import os
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

from gps_utils import (
    read_gps_table,
    compute_speed,
    add_time_tag,
    df_to_gdf_utm,
    load_polygon_union,
    filter_within_buffer,
    add_time_weights,
    make_grid_over_polygon,
    time_weighted_grid_split,
    selection_ratio_grid,
    kde_ud,
    isopleth_summary,
    compute_dispersion_timeseries,
)


BASE_OUT_DIR = os.environ.get(
    "GPS_ANALYSIS_OUT_DIR",
    os.path.join("gps_data_analysis", "gps_analysis_outputs"),
)
os.makedirs(BASE_OUT_DIR, exist_ok=True)


def to_perth_time(ts: pd.Series) -> pd.Series:
    """Convert naive/UTC timestamps to Australia/Perth (UTC+8)."""
    s = pd.to_datetime(ts, errors="coerce")
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert("Australia/Perth")


def run_paddock(cfg: Dict) -> None:
    """Run analysis for a single paddock config.

    Required keys in cfg:
      - label: output naming stem
      - gps_path: path to GPS CSV
      - paddock_path: WKT/Geo file path
      - filter_value: polygon attribute value (filter_col defaults to 'name')

    Optional keys:
      - out_dir (base): override BASE_OUT_DIR
      - filter_col: default 'name'
      - interval_min: default 5
      - tol_m: default 5.0
      - CELL_M: default 10
      - KDE_BW_M: default 25
      - GRID_RES: default 10
      - min_points_per_segment: default 50
      - export_dispersion: default True
    """

    label = cfg["label"]
    gps_path = cfg["gps_path"]
    paddock_path = cfg["paddock_path"]
    filter_value = cfg["filter_value"]
    filter_col = cfg.get("filter_col", "name")

    interval_min = int(cfg.get("interval_min", 5))
    tol_m = float(cfg.get("tol_m", 5.0))
    CELL_M = int(cfg.get("CELL_M", 10))
    KDE_BW_M = float(cfg.get("KDE_BW_M", 25))
    GRID_RES = int(cfg.get("GRID_RES", 10))
    min_pts = int(cfg.get("min_points_per_segment", 50))
    export_dispersion = bool(cfg.get("export_dispersion", True))

    out_base = cfg.get("out_dir", BASE_OUT_DIR)
    out_dir = os.path.join(out_base, label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== [{label}] Start ===")
    print(f"Input GPS: {gps_path}")
    print(f"Paddock: {paddock_path} (filter_col='{filter_col}', filter_value='{filter_value}')")
    print(f"Output dir: {out_dir}")

    # 1) Load GPS
    df = read_gps_table(
        gps_path,
        TIMESTAMP_COL_NAME="timestamp",
        LAT_COL_NAME="lat",
        LON_COL_NAME="lon",
        DEVICE_ID_COL_NAME="VID",
    )

    # 2) Timezone: convert to Perth (UTC+8)
    raw_first = pd.to_datetime(df["timestamp"], errors="coerce").dropna().iloc[0]
    df["timestamp"] = to_perth_time(df["timestamp"])  # tz-aware Perth time
    perth_first = df["timestamp"].dropna().iloc[0]
    print(f"Time example: raw='{raw_first}' -> Perth='{perth_first}' (+8h expected)")

    # 3) Speed + time tags
    ds = compute_speed(df)
    ds = add_time_tag(ds, interval_min=interval_min)

    # 4) Optional: herd dispersion time series
    if export_dispersion:
        dispersion_df = compute_dispersion_timeseries(
            ds, time_resolution_min=interval_min
        )
        # 常见质量阈值：仅保留 n>50 的时间片
        if "n" in dispersion_df.columns:
            dispersion_df = dispersion_df[dispersion_df["n"] > 50]
        disp_path = os.path.join(
            out_dir, f"{label}_Herd_Dispersion_{interval_min}min.csv"
        )
        dispersion_df.to_csv(disp_path, index=False)
        print(f"Saved dispersion: {disp_path}")

    # 5) Paddock utilization
    gdf = df_to_gdf_utm(ds)
    fence_union = load_polygon_union(
        paddock_path,
        target_crs=gdf.crs,
        filter_col=filter_col,
        filter_value=filter_value,
    )
    gdf_filt = filter_within_buffer(gdf, fence_union, tol_m=tol_m)
    gdf_w = add_time_weights(gdf_filt)

    # Grid + KDE parameters
    orig_grid = make_grid_over_polygon(fence_union, cell_m=CELL_M)

    rows_summary: List[Dict] = []
    processed_count, skipped_count = 0, 0
    for time_tag, grp in gdf_w.groupby("time_tag"):
        n_points = len(grp)
        if n_points < min_pts:
            skipped_count += 1
            continue

        tot_minutes = float(grp.get("dt_min", pd.Series(dtype=float)).sum())
        print(f"[time_tag {time_tag}] processed: points={n_points}, minutes≈{tot_minutes:.1f}")

        # Grid utilisation (time-weighted)
        grid = time_weighted_grid_split(grp, orig_grid)
        grid = selection_ratio_grid(grid, value_col="minutes")
        grid = grid[["cid", "minutes", "area_m2", "min_per_ha", "sel_ratio", "log_sel_ratio"]]

        grid_path = os.path.join(out_dir, f"{label}_GridUtilization_{time_tag}.csv")
        grid.to_csv(grid_path, index=False)

        # KDE UD + 50/95% isopleths
        xs, ys, z = kde_ud(grp, fence_union, bandwidth_m=KDE_BW_M, grid_res=GRID_RES)
        summ, _ = isopleth_summary(
            xs,
            ys,
            z,
            paddock_poly=fence_union,
            crs=getattr(grp, "crs", None),
            probs=(0.5, 0.95),
        )

        rows_summary.append(
            {
                "time_tag": time_tag,
                "n_points": n_points,
                "minutes_sum": tot_minutes,
                "paddock_area_ha": summ["paddock_area_ha"],
                "iso50_area_ha": summ[0.5]["area_ha"],
                "iso50_area_share": summ[0.5]["area_share_of_paddock"],
                "iso50_n_patches": summ[0.5]["n_patches"],
                "iso95_area_ha": summ[0.95]["area_ha"],
                "iso95_area_share": summ[0.95]["area_share_of_paddock"],
                "iso95_n_patches": summ[0.95]["n_patches"],
            }
        )
        processed_count += 1

    if rows_summary:
        df_summary = pd.DataFrame(rows_summary)
        summ_path = os.path.join(out_dir, f"{label}_Paddock_Utilization_Summary.csv")
        df_summary.to_csv(summ_path, index=False)
        print(f"Saved utilisation summary: {summ_path}")
    else:
        print("No summary rows produced (all segments skipped).")

    print(f"Segments: processed={processed_count}, skipped={skipped_count}")

    print(f"=== [{label}] Done ===\n")


def run_all(
    configs: List[Dict],
    base_outdir: str = BASE_OUT_DIR,
    parallel: bool = False,
    max_workers: int | None = None,
) -> None:
    if not parallel:
        for cfg in configs:
            cfg = {**cfg, "out_dir": base_outdir}
            run_paddock(cfg)
        return

    # Parallel execution
    workers = max_workers or min(len(configs), os.cpu_count() or 2)
    print(f"Launching {len(configs)} paddocks in parallel with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut2label = {}
        for cfg in configs:
            cfg = {**cfg, "out_dir": base_outdir}
            fut = ex.submit(run_paddock, cfg)
            fut2label[fut] = cfg.get("label", "<unknown>")
        for fut in as_completed(fut2label):
            label = fut2label[fut]
            try:
                fut.result()
                print(f"[parallel] Completed: {label}")
            except Exception as e:
                print(f"[parallel] FAILED: {label} :: {e}")


if __name__ == "__main__":
    # Default batch for Ridgefield 2024 Summer (Yellow/Green/Pink)
    CONFIGS = [
        {
            "label": "Ridgefield_2024_Summer_Yellow",
            "gps_path": "/home/ubuntu/work_dir/animal_data_analysis/GPS_Data_Analysis/Hotspot_Video_generation/Ridgefield_yellow_2024_summer.csv",
            "paddock_path": "/home/ubuntu/work_dir/animal_data_analysis/GPS_Data_Analysis/electronic_fence/Yellow_WKT.tsv",
            "filter_value": "Long Creek paddock",
            "interval_min": 30,
            # Optional overrides per paddock can be placed here
        },
        {
            "label": "Ridgefield_2024_Summer_Green",
            "gps_path": "/home/ubuntu/work_dir/animal_data_analysis/GPS_Data_Analysis/Hotspot_Video_generation/Ridgefield_green_2024_summer.csv",
            "paddock_path": "/home/ubuntu/work_dir/animal_data_analysis/GPS_Data_Analysis/electronic_fence/Green_WKT.tsv",
            "filter_value": "3000 Dam",
            "interval_min": 30,
        },
        {
            "label": "Ridgefield_2024_Summer_Pink",
            "gps_path": "/home/ubuntu/work_dir/animal_data_analysis/GPS_Data_Analysis/Hotspot_Video_generation/Ridgefield_pink_2024_summer.csv",
            "paddock_path": "/home/ubuntu/work_dir/animal_data_analysis/GPS_Data_Analysis/electronic_fence/Pink_WKT.tsv",
            "filter_value": "Walwalling Corner",
            "interval_min": 30,
        },
    ]

    # Switch to parallel execution by default (3 workers)
    run_all(CONFIGS, base_outdir=BASE_OUT_DIR, parallel=True, max_workers=3)
