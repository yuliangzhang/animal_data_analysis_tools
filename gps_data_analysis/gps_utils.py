import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import statsmodels.api as sm

import geopandas as gpd
from shapely.geometry import Point, Polygon
from sklearn.neighbors import KernelDensity
from shapely.geometry import Polygon, LineString
from shapely import wkt
from shapely.ops import unary_union


# ----------- 1) Read & prep GPS -----------
def read_gps_table(path: str,
                   TIMESTAMP_COL_NAME: str,
                   LAT_COL_NAME: str,
                   LON_COL_NAME: str,
                   DEVICE_ID_COL_NAME: str) -> pd.DataFrame:
    """"
    Read a GPS CSV/TSV file and return a cleaned DataFrame with standardized columns.
    Expects columns for timestamp (UTC), latitude, longitude, and device ID.
    Return the DataFrame with: columns renamed to 'timestamp', 'lat', 'lon', 'device_id' for the required columns.
    """

    # try to infer separator by extension; default to TSV as you said
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    # rename to our canonical names
    req = {TIMESTAMP_COL_NAME, LAT_COL_NAME, LON_COL_NAME, DEVICE_ID_COL_NAME}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in GPS file: {missing}")
    df = df.rename(columns={TIMESTAMP_COL_NAME: "timestamp",
                            LAT_COL_NAME: "lat",
                            LON_COL_NAME: "lon",
                            DEVICE_ID_COL_NAME: "device_id"})

    # parse UTC then convert to Perth
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # numeric safety
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    # drop impossible coords
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))].copy()
    # de-duplicate by id+timestamp
    df = df.drop_duplicates(subset=["device_id", "timestamp"]).reset_index(drop=True)

    return df


def utm_crs_for_lonlat(lon: float, lat: float) -> str:
    """
    Return a UTM EPSG code near the ranch center.
    Works in both hemispheres; Perth region is S hemisphere.
    """
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    return f"EPSG:{32700 + zone if south else 32600 + zone}"

#------------ 2) Compute speed & QC -----------
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in meters between two points."""
    R = 6371000.0
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = phi2 - phi1
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def compute_speed(df: pd.DataFrame, dist_thresh: float=300.0, speed_mps_thresh:float=5.0) -> pd.DataFrame:
    """
    Expect columns: timestamp, device_id, lat, lon.
    Returns df with tz-aware 'timestamp', 'dist_m', 'dt_s', 'speed_mps' per row.
    Handles gaps (dt from timestamps). Flags implausible jumps for QC.
    """
    d = df.copy()
    # d["timestamp"] = convert_time_tag_series(d["time_tag"], bin_ref="start", utc_offset_hours=None, tz=tz)
    d = d.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    # group-wise shift to get previous point
    d["lat_prev"] = d.groupby("device_id")["lat"].shift(1)
    d["lon_prev"] = d.groupby("device_id")["lon"].shift(1)
    d["t_prev"]   = d.groupby("device_id")["timestamp"].shift(1)

    # distance & dt
    mask = d["lat_prev"].notna()
    d.loc[mask, "dist_m"] = haversine_m(d.loc[mask, "lat_prev"], d.loc[mask, "lon_prev"],
                                        d.loc[mask, "lat"],      d.loc[mask, "lon"])
    d.loc[mask, "dt_s"] = (d.loc[mask, "timestamp"] - d.loc[mask, "t_prev"]).dt.total_seconds()
    d["speed_mps"] = d["dist_m"] / d["dt_s"]

    # basic QC flags
    d["jump_flag"] = (d["dist_m"] > dist_thresh) | (d["speed_mps"] > speed_mps_thresh)  # tune as needed

    return d

#------------ 3) Add Time Tag According to the required split frequency -----------

import pandas as pd
import numpy as np

def add_time_tag(df: pd.DataFrame,
                 interval_min: int = 5,
                 ts_col: str = "timestamp",
                 out_col: str = "time_tag") -> pd.DataFrame:
    """
    Add a per-day time segment label ("time_tag") in the format YYYY-MM-DD-XXXX (4 digits).

    Segments start at 00:00:00 of each day and are left-closed/right-open.
    Example for a 5-minute interval: 00:00:00–00:04:59.999 -> 0001; exactly 00:05:00 -> 0002.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    interval_min : int, default 5
        Segment width in minutes. For 5 minutes, there are 288 segments per day.
    ts_col : str, default "timestamp"
        Name of the timestamp column. Values will be coerced to pandas datetime.
        Works with tz-naive and tz-aware timestamps.
    out_col : str, default "time_tag"
        Name of the output column to store the segment label.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with an extra column `out_col`.
    """
    s = pd.to_datetime(df[ts_col], errors="coerce")

    # Seconds since the start of the day (00:00:00), preserving timezone if present
    secs_from_midnight = (s - s.dt.normalize()).dt.total_seconds()

    step = interval_min * 60  # segment length in seconds

    # Compute 1-based segment index; keep NA where timestamp is invalid
    idx = pd.Series(pd.NA, index=df.index, dtype="Int64")
    valid = s.notna()
    idx.loc[valid] = (secs_from_midnight.loc[valid] // step).astype(int) + 1

    # Compose YYYY-MM-DD-XXXX with zero-padded 4-digit index
    tag = pd.Series(pd.NA, index=df.index, dtype="string")
    tag.loc[valid] = s.loc[valid].dt.strftime("%Y-%m-%d-") + idx.loc[valid].astype(str).str.zfill(4)

    out = df.copy()
    out[out_col] = tag
    return out


#------------ 4) Calculate the dipsersion of herds based on the GPS data and specific time intervals -----------

def latlon_to_local_meters(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert lat/lon to local planar meters using equirectangular approximation
    around the dataset's median latitude/longitude. Accurate for local ranch scale.
    """
    lat0 = np.nanmedian(lat)
    lon0 = np.nanmedian(lon)
    # meters per degree approximations
    m_per_deg_lat = 111132.92
    m_per_deg_lon = 111412.84 * np.cos(np.deg2rad(lat0)) - 93.5 * np.cos(3*np.deg2rad(lat0))
    x = (lon - lon0) * m_per_deg_lon
    y = (lat - lat0) * m_per_deg_lat
    return x, y

def _convex_hull_area(points: np.ndarray) -> float:
    """
    Safe convex hull area for 2D points.
    In 2D, SciPy's ConvexHull.volume is the polygon area.
    Returns NaN if hull cannot be computed (e.g., <3 points or collinear).
    """
    if points is None or len(points) < 3:
        return np.nan
    try:
        hull = ConvexHull(points)
        return float(hull.volume)  # area in 2D
    except Exception:
        return np.nan

def per_time_dispersion(df_at_t: pd.DataFrame,
                        dbscan_eps_m: float = 40.0,
                        dbscan_min_samples: int = 10,
                        trim_q: float = 0.1) -> Dict:
    """
    Compute robust dispersion metrics for a single timestamp snapshot.
    Returns dict with:
      n, n_valid, centroid_x/y, rad_trim_mean, nnd_median, hull_area_m2 (if n>=3),
      dbscan_n_clusters, largest_cluster_ratio
    """
    sub = df_at_t[["device_id", "lat", "lon"]].dropna()
    n = len(sub)
    if n == 0:
        return {"n":0}

    # local xy in meters
    x, y = latlon_to_local_meters(sub["lat"].to_numpy(), sub["lon"].to_numpy())

    center_lat, center_lon = sub['lat'].mean(), sub['lon'].mean()
    avg_mv_dist_m = df_at_t['dist_m'].mean() if 'dist_m' in df_at_t.columns else np.nan
    avg_mv_speed_mps = df_at_t['speed_mps'].mean() if 'speed_mps' in df_at_t.columns else np.nan

    P = np.column_stack([x, y])
    n_valid = P.shape[0]

    # centroid & radius
    C = P.mean(axis=0)
    d2c = np.linalg.norm(P - C, axis=1)
    # trimmed mean radius
    if n_valid > 10:
        lo, hi = np.quantile(d2c, [trim_q, 1 - trim_q])
        mask_trim = (d2c >= lo) & (d2c <= hi)
        d2c_trim = d2c[mask_trim]
        rad_trim_mean = float(np.mean(d2c_trim)) if d2c_trim.size else float(np.mean(d2c))
    else:
        mask_trim = np.ones(n_valid, dtype=bool)
        rad_trim_mean = float(np.mean(d2c))

    # nearest neighbor distance median
    if n_valid >= 2:
        D = cdist(P, P)
        np.fill_diagonal(D, np.inf)
        nnd = np.min(D, axis=1)
        nnd_median = float(np.median(nnd))
    else:
        nnd_median = np.nan

    # ---- Convex hull areas (raw/trim) ----
    hull_area_raw  = _convex_hull_area(P)
    hull_area_trim = _convex_hull_area(P[mask_trim])

    # ---- DBSCAN clustering & largest cluster stats ----
    db_n_clusters, largest_ratio = 0, np.nan
    labels = np.full(n_valid, -1)
    try:
        db = DBSCAN(eps=dbscan_eps_m, min_samples=dbscan_min_samples).fit(P)
        labels = db.labels_
        valid = labels >= 0
        if valid.any():
            labs = labels[valid]
            counts = pd.Series(labs).value_counts()
            db_n_clusters = int(counts.size)
            top_lab = int(counts.idxmax())
            largest_ratio = float(counts.max() / n_valid)
            P_db = P[labels == top_lab]
        else:
            P_db = np.empty((0, 2))
    except Exception:
        P_db = np.empty((0, 2))

    hull_area_db = _convex_hull_area(P_db)

    return {
        "n": int(n),
        "n_valid": int(n_valid),
        "centroid_x": float(C[0]),
        "centroid_y": float(C[1]),
        "rad_trim_mean_m": rad_trim_mean,
        "nnd_median_m": nnd_median,
        "hull_area_m2_raw": hull_area_raw,
        "hull_area_m2_trim": hull_area_trim,   # robust to outliers via trimming
        "hull_area_m2_db": hull_area_db,       # robust via DBSCAN largest cluster
        "dbscan_n_clusters": db_n_clusters,
        "largest_cluster_ratio": largest_ratio,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "avg_mv_dist_m": avg_mv_dist_m,
        "avg_mv_speed_mps": avg_mv_speed_mps
    }

def compute_dispersion_timeseries(d: pd.DataFrame,
                                  dbscan_eps_m: float = 40.0,
                                  dbscan_min_samples: int = 10,
                                  trim_q: float = 0.15,
                                  time_resolution_min: int = 5) -> pd.DataFrame:
    """
    Group by timestamp and compute dispersion metrics for each time step.
    Requires columns: timestamp (tz-aware), SHEEP_VID, lat, lon.
    """
    # ensure timestamp present
    if "time_tag" not in d.columns:
        d = add_time_tag(d, interval_min=time_resolution_min)
    out = []
    for ts, g in d.groupby("time_tag", sort=True):
        m = per_time_dispersion(g, dbscan_eps_m, dbscan_min_samples, trim_q)
        m["time_tag"] = ts
        out.append(m)
    return pd.DataFrame(out).sort_values("time_tag")

#------------ 5) Paddock Utilization Analysis Helper -----------

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
        # Build geometry from WKT; set CRS to WGS84
        df["geometry"] = df[wkt_col].astype(str).apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

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
    # geopandas compatibility: prefer union_all if available, else unary_union
    if hasattr(sel, "union_all"):
        union_poly = sel.union_all()
    else:
        union_poly = unary_union(sel.geometry)
    if union_poly.is_empty:
        raise ValueError("Selected polygons union is empty. Check geometry validity.")
    return union_poly


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


def add_time_weights(gdf_pts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute minutes between consecutive fixes per animal.
    Use median dt for last points; cap extreme gaps at 3×median.
    """
    d = gdf_pts.sort_values(["device_id", "timestamp"]).copy()
    d["t_next"] = d.groupby("device_id")["timestamp"].shift(-1)
    dt = (d["t_next"] - d["timestamp"]).dt.total_seconds()/60.0
    med = float(np.nanmedian(dt))
    dt = dt.fillna(med).clip(0, 3*med if med > 0 else None)
    d["dt_min"] = dt
    return d.drop(columns=["t_next"])


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

def time_weighted_grid_split(gdf_pts: gpd.GeoDataFrame, orig_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    grid = orig_grid.copy()
    grid = grid.set_crs(gdf_pts.crs)
    # spatial join
    frame = gpd.sjoin(gdf_pts, grid, predicate="within")[["cid", "dt_min", "time_tag", "timestamp"]]
    util = frame.groupby("cid")["dt_min"].sum().reset_index().rename(columns={"dt_min": "minutes"})
    grid = grid.merge(util, on="cid", how="left").fillna({"minutes": 0.0})
    grid["area_m2"] = grid.area
    grid["min_per_ha"] = grid["minutes"] / (grid["area_m2"]/10000.0 + 1e-9)
    return grid


def selection_ratio_grid(grid: gpd.GeoDataFrame, value_col="minutes") -> gpd.GeoDataFrame:
    tshare = grid[value_col] / grid[value_col].sum()
    ashare = grid["area_m2"] / grid["area_m2"].sum()
    w = (tshare / (ashare + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0)
    grid["sel_ratio"] = w
    grid["log_sel_ratio"] = np.log1p(w)  # log(1+w), helps visual scaling
    return grid



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
            # We requested a single level, so use index 0
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
        # Fall back silently to collection paths if available
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
