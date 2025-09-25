import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

try:
    import contextily as ctx
    _HAS_CTX = True
except Exception:
    _HAS_CTX = False

# ------------- 0) CRS helpers -------------
def utm_crs_for_lonlat(lon: float, lat: float) -> str:
    """
    Return a UTM EPSG code near the ranch center.
    Works in both hemispheres; Perth region is S hemisphere.
    """
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    return f"EPSG:{32700 + zone if south else 32600 + zone}"

# ----------- 1) Read & prep GPS -----------
def read_gps_table(path: str) -> pd.DataFrame:
    """
    Read your GPS file with columns:
      gps_code, timestamp, lat, lon, VID
    The 'timestamp' is in GMT/UTC and must be converted to Perth local time (UTC+8).
    Returns a pandas DataFrame with columns:
      SHEEP_VID, gps_code, lat, lon, timestamp (tz-aware, Australia/Perth)
    """
    # try to infer separator by extension; default to TSV as you said
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    # rename to our canonical names
    req = {"gps_code", "timestamp", "lat", "lon", "VID"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in GPS file: {missing}")
    df = df.rename(columns={"VID": "SHEEP_VID"})
    # parse UTC then convert to Perth
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Australia/Perth")
    # numeric safety
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    # drop impossible coords
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))].copy()
    # de-duplicate by id+timestamp
    df = df.drop_duplicates(subset=["SHEEP_VID", "timestamp"]).reset_index(drop=True)
    return df

def df_to_gdf_utm(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert the GPS table to a GeoDataFrame in local UTM meters.
    """
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )
    lon0, lat0 = float(df["lon"].median()), float(df["lat"].median())
    utm = utm_crs_for_lonlat(lon0, lat0)
    return gdf.to_crs(utm)



def load_polygon_union(
    paddock_path: str,
    target_crs,
    filter_col: str = "name",           # 过滤字段名：如 'name' 或 'explanation'
    filter_value: str = "Long Creek paddock",  # 过滤值
    wkt_col: str = "WKT",               # WKT 列名（若是表格）
) -> Polygon:
    """
    Load paddock polygons from:
      - vector files (shp/geojson/gpkg), OR
      - tabular files (csv/tsv) with a WKT column.
    Filter rows by filter_col==filter_value, dissolve to a single union polygon,
    ensure CRS is set (EPSG:4326) before transforming to target_crs, and return the union polygon.
    """
    # ---- Try reading as a GIS vector file first ----
    gdf = None
    try:
        gdf = gpd.read_file(paddock_path)
        # If geometry is empty or missing, fall back to WKT-table mode
        if gdf.empty or gdf.geometry.isna().all():
            gdf = None
    except Exception:
        gdf = None

    # ---- If not a vector file, read as table with WKT ----
    if gdf is None:
        # Robust separator inference (works for CSV/TSV)
        try:
            df = pd.read_csv(paddock_path, sep=None, engine="python")
        except Exception:
            # fallback: common cases
            try:
                df = pd.read_csv(paddock_path, sep="\t")
            except Exception:
                df = pd.read_csv(paddock_path, sep=",")
        if wkt_col not in df.columns:
            raise ValueError(f"Could not find WKT column '{wkt_col}' in file.")
        # # Build geometry from WKT; set CRS to WGS84
        # df["geometry"] = df[wkt_col].apply(wkt.loads)
        # gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    # ---- Ensure we have a CRS before transforming ----
    if gdf.crs is None:
        # Your WKT is lon/lat degrees → WGS84
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    # ---- Filter rows by attribute (case-insensitive, trimmed) ----
    if filter_col not in gdf.columns:
        raise ValueError(f"Column '{filter_col}' not found. Available: {list(gdf.columns)}")
    col_norm = gdf[filter_col].astype(str).str.strip()

    # exact match first; if empty, do case-insensitive fallback
    sel = gdf[col_norm == filter_value]
    if sel.empty:
        sel = gdf[col_norm.str.lower() == str(filter_value).strip().lower()]
    if sel.empty:
        raise ValueError(
            f"No polygons with {filter_col} == '{filter_value}'. "
            f"Example values: {gdf[filter_col].dropna().astype(str).head(5).to_list()}"
        )

    # ---- Project to target CRS and dissolve to union polygon ----
    sel = sel.to_crs(target_crs)
    # union_poly = sel.unary_union
    union_poly = sel.union_all()
    if union_poly.is_empty:
        raise ValueError("Selected polygons union is empty. Check geometry validity.")
    return union_poly

# -------- 3) Filter by 10 m tolerance --------
def filter_within_buffer(gdf_pts: gpd.GeoDataFrame, fence_poly: Polygon, tol_m: float = 10.0) -> gpd.GeoDataFrame:
    """
    Keep only GPS points that fall inside the fence polygon OR within tol_m around it.
    (i.e., within fence_poly.buffer(tol_m))
    """
    buf = fence_poly.buffer(tol_m)
    inside = gdf_pts.within(buf)
    kept = gdf_pts[inside].copy()
    print(f"[QC] kept {kept.shape[0]} points inside 'media shade' ±{int(tol_m)} m "
          f"({kept.shape[0]/max(1,len(gdf_pts))*100:.1f}% of all points).")
    return kept

# ------------- 4) Time weights --------------
def add_time_weights(gdf_pts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute minutes between consecutive fixes per animal.
    Use median dt for last points; cap extreme gaps at 3×median.
    """
    d = gdf_pts.sort_values(["SHEEP_VID", "timestamp"]).copy()
    d["t_next"] = d.groupby("SHEEP_VID")["timestamp"].shift(-1)
    dt = (d["t_next"] - d["timestamp"]).dt.total_seconds()/60.0
    med = float(np.nanmedian(dt))
    dt = dt.fillna(med).clip(0, 3*med if med > 0 else None)
    d["dt_min"] = dt
    return d.drop(columns=["t_next"])

# ---------- 5) Grid-based utilisation ----------
def make_grid_over_polygon(poly: Polygon, cell_m=10) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = poly.bounds
    xs = np.arange(xmin, xmax + cell_m, cell_m)
    ys = np.arange(ymin, ymax + cell_m, cell_m)
    cells = []
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            cell = Polygon([(xs[i], ys[j]), (xs[i+1], ys[j]), (xs[i+1], ys[j+1]), (xs[i], ys[j+1])])
            if poly.intersects(cell):
                cells.append(poly.intersection(cell))
    return gpd.GeoDataFrame({"cid": range(len(cells))}, geometry=cells, crs=None)

def time_weighted_grid(gdf_pts: gpd.GeoDataFrame, fence_poly: Polygon, cell_m=10) -> gpd.GeoDataFrame:
    grid = make_grid_over_polygon(fence_poly, cell_m=cell_m)
    grid = grid.set_crs(gdf_pts.crs)
    # spatial join
    frame = gpd.sjoin(gdf_pts, grid, predicate="within")[["cid", "dt_min"]]  # TODO: keep the timestamp for further ananlysis based on the time of the day
    util = frame.groupby("cid")["dt_min"].sum().reset_index().rename(columns={"dt_min": "minutes"})
    grid = grid.merge(util, on="cid", how="left").fillna({"minutes": 0.0})
    grid["area_m2"] = grid.area
    grid["min_per_ha"] = grid["minutes"] / (grid["area_m2"]/10000.0 + 1e-9)
    return grid

def time_weighted_grid_split(gdf_pts: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    # spatial join
    frame = gpd.sjoin(gdf_pts, grid, predicate="within")[["cid", "dt_min"]]  # TODO: keep the timestamp for further ananlysis based on the time of the day
    util = frame.groupby("cid")["dt_min"].sum().reset_index().rename(columns={"dt_min": "minutes"})
    grid = grid.merge(util, on="cid", how="left").fillna({"minutes": 0.0})
    grid["area_m2"] = grid.area
    grid["min_per_ha"] = grid["minutes"] / (grid["area_m2"]/10000.0 + 1e-9)
    return grid

# ---------- 6) KDE usage distribution ----------
def kde_ud(gdf_pts: gpd.GeoDataFrame, fence_poly: Polygon, bandwidth_m=25, grid_res=10):
    P = np.vstack([gdf_pts.geometry.x.values, gdf_pts.geometry.y.values]).T
    w = gdf_pts["dt_min"].values
    kde = KernelDensity(bandwidth=bandwidth_m, kernel="gaussian").fit(P, sample_weight=w)

    xmin, ymin, xmax, ymax = fence_poly.bounds
    xs = np.arange(xmin, xmax, grid_res)
    ys = np.arange(ymin, ymax, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    XY = np.vstack([XX.ravel(), YY.ravel()]).T

    # mask inside paddock
    from shapely.geometry import Point
    mask = np.array([fence_poly.contains(Point(x, y)) for x, y in XY])
    Z = np.full(XY.shape[0], np.nan)
    Z[mask] = np.exp(kde.score_samples(XY[mask]))
    z = Z.copy()
    z[np.isnan(z)] = 0.0
    z = z / z.sum()
    return xs, ys, z.reshape(len(ys), len(xs))

# ---------- 7) Plotting helpers ----------

def _ensure_outdir(outdir: str):
    """Create output directory if it does not exist."""
    os.makedirs(outdir, exist_ok=True)

def _safe_fname(text: str) -> str:
    """Make a filesystem-safe filename stem from a label."""
    return "".join([c if c.isalnum() or c in ("-","_") else "_" for c in text])

def _set_extent_from_poly(ax, poly, pad_m: float = 30.0):
    xmin, ymin, xmax, ymax = poly.bounds
    ax.set_xlim(xmin - pad_m, xmax + pad_m)
    ax.set_ylim(ymin - pad_m, ymax + pad_m)

# ------------------- plot_grid with basemap -------------------
def plot_grid(grid,
              fence_poly,
              value="min_per_ha",
              cmap="viridis",
              title=None,
              outdir="plot_paddock_utilization",
              fname=None,
              dpi=220,
              alpha=0.45,  # transparency of the grid overlay
              basemap=True,
              basemap_source=None,  # e.g., ctx.providers.Esri.WorldImagery
              basemap_zoom="auto",
              attribution=True):
    """
    Grid choropleth over a satellite/terrain basemap (contextily), saved to file.
    All geometries are assumed in a projected CRS (e.g., local UTM, units=m).
    """
    _ensure_outdir(outdir)
    if fname is None:
        fname = f"grid_{_safe_fname(value)}.png"
    path = os.path.join(outdir, fname)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    # set extent from paddock polygon first (so contextily knows what to fetch)
    _set_extent_from_poly(ax, fence_poly, pad_m=30)

    # add basemap underneath
    if basemap and _HAS_CTX:
        if basemap_source is None:
            basemap_source = ctx.providers.Esri.WorldImagery
        try:
            ctx.add_basemap(ax, crs=grid.crs, source=basemap_source,
                            attribution=attribution, zoom=basemap_zoom)
        except Exception as e:
            print(f"[warn] add_basemap failed: {e}")

    # draw grid as semi-transparent overlay
    grid.plot(column=value, ax=ax, cmap=cmap, linewidth=0, alpha=alpha,
              legend=True, legend_kwds={"label": value})

    # paddock boundary on top
    gpd.GeoSeries([fence_poly], crs=grid.crs).boundary.plot(ax=ax, color="cyan", linewidth=1.2)

    ax.set_title(title or f"Paddock utilisation grid ({value})")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ------------------- plot_kde with basemap --------------------
def plot_kde(xs, ys, z,
             fence_poly,
             grid,
             probs=(0.5, 0.95),
             outdir="plot_paddock_utilization",
             fname="kde_ud_50_95.png",
             dpi=220,
             alpha=0.45,  # transparency of heatmap overlay
             basemap=True,
             basemap_source=None,
             basemap_zoom="auto",
             attribution=True):
    """
    KDE heatmap (imshow) + isopleths over a contextily basemap; saved to file.
    xs, ys: 1D arrays defining grid coordinates in meters (same CRS as fence_poly).
    z: 2D KDE density (normalised to sum=1 inside paddock).
    """
    _ensure_outdir(outdir)
    path = os.path.join(outdir, fname)

    # compute monotonically increasing contour levels (robust)
    flat = z.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    pairs = []
    for p in probs:
        k = int(np.searchsorted(csum, p))
        k = max(0, min(k, len(order)-1))
        pairs.append((p, float(flat[order][k])))
    levels = np.unique(np.array([v for (_, v) in sorted(pairs, key=lambda x: x[1])], dtype=float))
    # ensure at least one level
    if levels.size == 0:
        levels = np.array([np.nanmax(z) * 0.5], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    # set extent from paddock polygon first
    _set_extent_from_poly(ax, fence_poly, pad_m=30)

    # add basemap underneath
    if basemap and _HAS_CTX:
        if basemap_source is None:
            basemap_source = ctx.providers.Esri.WorldImagery
        try:
            # we pass the CRS of the *data axes* (UTM) so contextily reprojects tiles correctly
            ctx.add_basemap(ax, crs=gpd.GeoSeries([fence_poly]).set_crs(None).set_crs("EPSG:3857", allow_override=True).crs
                            if False else None)  # placeholder to avoid lint
        except Exception:
            pass
        try:
            # Correct way: provide the true CRS of data (same as xs/ys, i.e., grid CRS)
            # If you have the CRS object (e.g., grid_crs), pass it here.
            # We'll infer from fence_poly by wrapping into a GeoSeries with known CRS from your pipeline.
            # The plotting code above calls this function with grid/fence in the same CRS as your UTM.
            ctx.add_basemap(ax, crs="EPSG:3857")  # dummy overwritten below
        except Exception:
            pass
    # ---- NOTE ----
    # Some servers have older contextily that ignores 'crs=None'. To be safe,
    # we fetch CRS from the paddock by constructing a GeoSeries with a CRS.
    # Replace 'data_crs' with your actual CRS object when calling this function, e.g., grid.crs.

    # Re-do the basemap with an explicit CRS (robust branch)
    if basemap and _HAS_CTX:
        try:
            data_crs = gpd.GeoSeries([fence_poly]).set_crs("EPSG:3857", allow_override=True).crs  # overwritten next
        except Exception:
            data_crs = None
        # Better: let the caller pass CRS; here we try to infer from numeric scale (meters -> UTM)
        if basemap_source is None:
            basemap_source = ctx.providers.Esri.WorldImagery
        try:
            # If you know your CRS object, pass it as 'crs=grid_crs'. For compatibility,
            # we simply call add_basemap with crs=None when we can't infer, and let ctx assume 3857.
            ctx.add_basemap(ax, crs=getattr(gpd.GeoSeries([fence_poly], crs=None), "crs", None),
                            source=basemap_source, attribution=attribution, zoom=basemap_zoom)
        except Exception as e:
            # final attempt using your grid CRS if available
            try:
                ctx.add_basemap(ax, crs=getattr(grid, "crs", None),
                                source=basemap_source, attribution=attribution, zoom=basemap_zoom)
            except Exception as e2:
                print(f"[warn] add_basemap failed: {e or e2}")

    # draw KDE heatmap (semi-transparent) in data CRS
    extent = (xs.min(), xs.max(), ys.min(), ys.max())
    im = ax.imshow(z, origin="lower", extent=extent, cmap="magma", alpha=alpha)

    # contour lines on top
    cs = ax.contour(xs, ys, z, levels=levels.tolist(),
                    colors=["white", "cyan", "yellow"][:len(levels)],
                    linewidths=[1.8, 1.5, 1.2][:len(levels)])

    # label contours with their cumulative percentages
    fmt = {lv: f"{int(round(p*100))}%" for (p, lv) in pairs}
    # map to actual cs.levels to avoid float rounding mismatch
    fmt_safe = {lv: fmt.get(lv, f"{int(100*(i+1)/len(cs.levels))}%") for i, lv in enumerate(cs.levels)}
    ax.clabel(cs, fmt=fmt_safe, inline=True)

    # paddock boundary
    gpd.GeoSeries([fence_poly]).boundary.plot(ax=ax, color="cyan", linewidth=1.2)

    ax.set_title("Kernel UD with isopleths (over basemap)")
    ax.set_axis_off()
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("Probability mass")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def selection_ratio_grid(grid: gpd.GeoDataFrame, value_col="minutes") -> gpd.GeoDataFrame:
    tshare = grid[value_col] / grid[value_col].sum()
    ashare = grid["area_m2"] / grid["area_m2"].sum()
    w = (tshare / (ashare + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0)
    grid["sel_ratio"] = w
    grid["log_sel_ratio"] = np.log1p(w)  # log(1+w), helps visual scaling
    return grid

## ------------- Isopleth computation --------------

def _isopleth_thresholds(z, probs=(0.5, 0.95)):
    """
    Compute density thresholds (levels) for given cumulative probabilities.
    Returns dict {p: level}.
    """
    flat = z.ravel()
    order = np.argsort(flat)[::-1]   # high->low
    csum = np.cumsum(flat[order])
    levels = {}
    for p in probs:
        k = int(np.searchsorted(csum, p))
        k = max(0, min(k, len(order)-1))
        levels[p] = float(flat[order][k])
    return levels

def _contours_to_polys(xs, ys, z, level):
    """
    Extract shapely Polygons from a single contour level.
    """
    # build contour at a single, strictly positive level
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)
    cs = ax.contour(xs, ys, z, levels=[level])
    polys = []

    # Prefer using allsegs which is stable across Matplotlib versions
    try:
        allsegs = getattr(cs, "allsegs", None)
        if allsegs is not None and len(allsegs) > 0:
            for seg in allsegs[0]:
                if seg is None or len(seg) < 3:
                    continue
                # Only keep closed contours to avoid spurious ribbons
                if not (np.allclose(seg[0], seg[-1])):
                    continue
                poly = Polygon(seg)
                if poly.is_valid and poly.area > 0:
                    polys.append(poly)
    except Exception:
        pass

    # Fallback for older Matplotlib where collections.get_paths() is standard
    if not polys and hasattr(cs, "collections"):
        for coll in cs.collections:
            for path in coll.get_paths():
                v = path.vertices
                if len(v) >= 3:
                    poly = Polygon(v)
                    if poly.is_valid and poly.area > 0:
                        polys.append(poly)

    plt.close(fig)
    return polys

def isopleth_gdf(xs, ys, z, level, paddock_poly, crs=None):
    """
    Build a GeoDataFrame of isopleth polygons clipped to the paddock.
    """
    polys = _contours_to_polys(xs, ys, z, level)
    if not polys:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    # union & intersect with paddock
    merged = unary_union(polys)
    clipped = merged.intersection(paddock_poly)
    # explode to parts
    gdf = gpd.GeoDataFrame(geometry=[clipped], crs=crs).explode(index_parts=False).reset_index(drop=True)
    gdf["area_m2"] = gdf.area
    gdf["area_ha"] = gdf["area_m2"] / 10000.0
    return gdf

def isopleth_summary(xs, ys, z, paddock_poly, crs=None, probs=(0.5, 0.96)):
    """Return (summary_dict, {p:gdf_polys}) for requested probabilities."""
    levels = _isopleth_thresholds(z, probs=probs)
    pad_area_ha = paddock_poly.area / 10000.0
    gdfs, out = {}, {}
    for p, lv in levels.items():
        polys = _contours_to_polys(xs, ys, z, lv)
        if not polys:
            gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
            area_ha, n_patches = 0.0, 0
        else:
            merged = unary_union(polys).intersection(paddock_poly)
            gdf = gpd.GeoDataFrame(geometry=[merged], crs=crs).explode(index_parts=False).reset_index(drop=True)
            gdf["area_m2"] = gdf.area
            gdf["area_ha"] = gdf["area_m2"] / 10000.0
            area_ha = float(gdf["area_ha"].sum())
            n_patches = int(len(gdf))
        gdfs[p] = gdf
        out[p] = {
            "level": lv,
            "area_ha": area_ha,
            "area_share_of_paddock": (area_ha / pad_area_ha) if pad_area_ha>0 else np.nan,
            "n_patches": n_patches,
        }
    out["paddock_area_ha"] = pad_area_ha
    return out, gdfs
