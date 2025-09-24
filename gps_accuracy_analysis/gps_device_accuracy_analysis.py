# -*- coding: utf-8 -*-
"""
Multi-device GPS accuracy analysis with your original visualization integrated.

Directory layout:
    ROOT_DIR/
        GPS171/
            GPS0000_250910.txt
            GPS0000_250911.txt
            ...
        GPS332/
            GPS0000_250911.txt
            GPS0000_250912.txt
            ...

Device name = subfolder name (e.g., 'GPS171').

File format:
- Either with a header containing "latitude"/"Hours" etc., or a raw 10-column CSV:
  Hours, minutes, seconds, latitude, longitude, altitude, speed, course, HDOP, satellites

Key features:
- Use average lat/lon as temporary "true" location (per device).
- Compute Haversine error (your approach), filter long-tail by a configurable threshold.
- Recenter with the filtered mean (as in your code), then recompute errors.
- Compute horizontal metrics: Mean/Max/Std, CEP50/CEP95, 2DRMS, RMS_H, avg_r (mean radial ENU).
- Per-day and overall summaries.
- HDOP quality subsets in parallel: thresholds [250, 300, 350, 400, 450, 500].
- Visualizations: (1) error histogram; (2) GPS points vs "true" location; (3) HDOP vs error.
- Save error details and metric summaries per device.

All comments in English as requested.
"""

import os, re, math
from math import radians, sin, cos, atan2, sqrt
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- Configuration -----------------------------
# Resolve paths relative to this script to avoid CWD issues
THIS_DIR = Path(__file__).parent
ROOT_DIR = THIS_DIR / "device_data"   # root folder that contains per-device subfolders
OUT_ROOT = THIS_DIR / "gps_analysis_res"  # where per-device outputs go

# File name pattern inside each device folder
FILE_GLOB = "GPS0000_*.txt"

# Long-tail error filter (meters) before recomputing the mean "true" location
ERROR_FILTER_M = 150.0

# HDOP threshold list for subsets (<= threshold). Use values like [250, 300, 350, 400, 450, 500]
HDOP_THRESHOLDS = [250, 300, 350, 400, 450, 500]

# Histogram bins
HIST_BINS = 50

# -------------------------------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_gps_file(path: Path) -> pd.DataFrame:
    """
    Read a GPS text/CSV file that may or may not have a header.
    If no header, assume:
        Hours, minutes, seconds, latitude, longitude, altitude, speed, course, HDOP, satellites
    """
    cols = ["Hours","minutes","seconds","latitude","longitude","altitude",
            "speed","course","HDOP","satellites"]
    head = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    has_header = (len(head)>0) and (("latitude" in head[0]) or ("Hours" in head[0]))
    df = pd.read_csv(path, header=0 if has_header else None, names=cols)

    # Normalize numeric types and drop zero fixes
    for c in ["latitude","longitude","altitude","HDOP","satellites"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Scale HDOP by 100 as per analysis convention
    if "HDOP" in df.columns:
        df["HDOP"] = df["HDOP"] * 100.0

    # Filter out rows where either latitude or longitude is zero
    df["is_zero"] = (df["latitude"] == 0) | (df["longitude"] == 0)
    df = df[~df["is_zero"]].copy()

    # Derive day from filename: GPS0000_YYMMDD.txt
    m = re.search(r"_(\d{6})", path.name)
    df["day"] = m.group(1) if m else "unknown"

    # Seconds-of-day (optional)
    for c in ["Hours","minutes","seconds"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["t_s"] = df["Hours"]*3600 + df["minutes"]*60 + df["seconds"]

    return df

# ----------------------------- Your core math -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (meters) between two (lat,lon) in degrees."""
    R = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlmb/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def enu_from_mean(df: pd.DataFrame):
    """
    Convert to local ENU (E,N only) relative to mean( lat, lon ).
    Return a copy with dx_m, dy_m, horiz_err_m, plus center in attrs.
    """
    R = 6371000.0
    lat0 = df["LAT"].mean()
    lon0 = df["LON"].mean()
    lat0_rad = math.radians(lat0)

    dx = (df["LON"] - lon0) * (math.cos(lat0_rad)) * (R * math.pi/180.0)
    dy = (df["LAT"] - lat0) * (R * math.pi/180.0)

    out = df.copy()
    out["dx_m"], out["dy_m"] = dx, dy
    out["horiz_err_m"] = np.hypot(dx, dy)
    out.attrs["center"] = {"lat_mean": lat0, "lon_mean": lon0}
    return out

# ----------------------------- Metrics -----------------------------
def metrics_from_errors(err_m: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> Dict[str, float]:
    """Compute robust horizontal metrics."""
    n = len(err_m)
    if n == 0:
        return {k: np.nan for k in [
            "n_fixes","Mean_Error_m","Max_Error_m","Std_Error_m",
            "CEP50_m","CEP95_m","RMS_H_m","avg_r_m","2DRMS_m",
            "std_dx_m","std_dy_m"
        ]}
    varx = np.var(dx, ddof=1) if n>1 else 0.0
    vary = np.var(dy, ddof=1) if n>1 else 0.0
    out = {
        "n_fixes": n,
        # Your original trio
        "Mean_Error_m": float(np.mean(err_m)),
        "Max_Error_m": float(np.max(err_m)),
        "Std_Error_m": float(np.std(err_m, ddof=1)) if n>1 else 0.0,
        # Percentiles
        "CEP50_m": float(np.percentile(err_m, 50)),
        "CEP95_m": float(np.percentile(err_m, 95)),
        # RMS & avg radial ENU
        "RMS_H_m": float(np.sqrt(np.mean(dx**2 + dy**2))),
        "avg_r_m": float(np.mean(np.sqrt(dx**2 + dy**2))),
        # 2DRMS uses variances of orthogonal axes
        "2DRMS_m": float(2.0*np.sqrt(varx + vary)) if n>1 else np.nan,
        # Directional std
        "std_dx_m": float(np.std(dx, ddof=1)) if n>1 else np.nan,
        "std_dy_m": float(np.std(dy, ddof=1)) if n>1 else np.nan,
    }
    return out

# ----------------------------- Visualization (integrated) -----------------------------
def plot_error_histogram(df_err: pd.DataFrame, out_png: Path, title: str):
    """Your histogram style."""
    plt.figure(figsize=(8,5))
    plt.hist(df_err["Error_m"], bins=HIST_BINS, edgecolor='black')
    plt.title(title)
    plt.xlabel("Error (meters)")
    plt.ylabel("Number of Points")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_scatter_vs_true(df_err: pd.DataFrame, true_lat: float, true_lon: float, out_png: Path, title: str):
    """Your scatter style: GPS points vs 'true' mean."""
    plt.figure(figsize=(8,8))
    plt.scatter(df_err["LON"], df_err["LAT"], c='blue', s=10, label='GPS Points')
    plt.scatter(true_lon, true_lat, c='red', marker='x', s=100, label='Mean Center')
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_hdop_vs_error(df_err: pd.DataFrame, out_png: Path, title: str):
    """Additional scatter: HDOP vs error."""
    if "HDOP" not in df_err.columns: 
        return
    plt.figure(figsize=(8,5))
    plt.scatter(df_err["HDOP"], df_err["Error_m"], s=10, alpha=0.5)
    plt.title(title)
    plt.xlabel("HDOP")
    plt.ylabel("Error (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_hdop_error_boxplot(df_err: pd.DataFrame, out_png: Path, title: str,
                            thresholds: List[int] = [100, 200, 300, 400, 500]):
    """Boxplot of Error vs cumulative HDOP thresholds.

    Groups shown: <=100, <=200, <=300, <=400, <=500, >500.
    Each threshold group is cumulative (e.g., <=300 includes <=200 and <=100).
    """
    if "HDOP" not in df_err.columns:
        return
    errs = []
    labels = []
    for thr in thresholds:
        sel = df_err[df_err["HDOP"] <= thr]
        if len(sel) > 0:
            errs.append(sel["Error_m"].to_numpy())
            labels.append(f"<= {thr}")
    # > max threshold group
    sel_gt = df_err[df_err["HDOP"] > thresholds[-1]]
    if len(sel_gt) > 0:
        errs.append(sel_gt["Error_m"].to_numpy())
        labels.append(f"> {thresholds[-1]}")

    if not errs:
        return

    plt.figure(figsize=(9,5))
    # use tick_labels to avoid Matplotlib 3.9 deprecation warning
    plt.boxplot(errs, tick_labels=labels, showfliers=True)
    plt.title(title)
    plt.xlabel("HDOP threshold (x100)")
    plt.ylabel("Error (m)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ----------------------------- Per-device pipeline -----------------------------
def process_device_folder(device_dir: Path, out_root: Path,
                          error_filter_m: float = ERROR_FILTER_M,
                          hdop_thresholds: List[float] = HDOP_THRESHOLDS):
    """
    Read all files under one device folder, compute metrics and plots.
    Save:
      device_overall.csv, device_perday.csv, device_centers.csv,
      gps_error_data.csv, gps_error_summary.csv,
      plots/*.png
    """
    device_id = device_dir.name
    out_dir = out_root / device_id
    plots_dir = out_dir / "plots"
    ensure_dir(out_dir); ensure_dir(plots_dir)

    # Load files
    files = sorted(device_dir.glob(FILE_GLOB))
    if not files:
        print(f"[WARN] No files for device {device_id}")
        return

    frames = []
    for p in files:
        try:
            df = read_gps_file(p)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")

    if not frames:
        print(f"[WARN] No valid data for {device_id}")
        return

    raw = pd.concat(frames, ignore_index=True)

    # Normalize to your column names for downstream functions
    raw["LAT"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["LON"] = pd.to_numeric(raw["longitude"], errors="coerce")

    # ---- Step 1: first-pass mean center (temporary truth) using ALL valid points ----
    true_lat_1 = raw["LAT"].mean()
    true_lon_1 = raw["LON"].mean()

    # ---- Step 2: compute Haversine errors to the first-pass center ----
    raw["Error_m"] = raw.apply(lambda r: haversine(true_lat_1, true_lon_1, r["LAT"], r["LON"]), axis=1)

    # ---- Step 3: filter long-tail (your approach), then recompute the center ----
    filt = raw[raw["Error_m"] < error_filter_m].copy()
    true_lat = filt["LAT"].mean()
    true_lon = filt["LON"].mean()

    # Recompute errors to the filtered center
    filt["Error_m"] = filt.apply(lambda r: haversine(true_lat, true_lon, r["LAT"], r["LON"]), axis=1)

    # ---- Step 4: ENU errors (for RMS/2DRMS etc.) relative to filtered mean ----
    # Prepare a slim DF for ENU conversion
    slim = filt[["LAT","LON","HDOP","satellites","day","t_s"]].copy()
    enu = enu_from_mean(slim)  # adds dx_m, dy_m, horiz_err_m
    # Attach back to filt for unified export
    filt = filt.join(enu[["dx_m","dy_m","horiz_err_m"]])

    # ---- Step 5: Per-day & overall metrics for RAW (after long-tail filter) ----
    # (i.e., this "RAW" now means after your Error<150m step; matches your intent)
    overall_records = []
    perday_records = []

    # Overall (no HDOP filter)
    m_all = metrics_from_errors(filt["Error_m"].to_numpy(),
                                filt["dx_m"].to_numpy(),
                                filt["dy_m"].to_numpy())
    m_all.update({"device_id": device_id, "subset": "raw"})
    overall_records.append(m_all)

    # Per-day
    for day, g in filt.groupby("day"):
        md = metrics_from_errors(g["Error_m"].to_numpy(), g["dx_m"].to_numpy(), g["dy_m"].to_numpy())
        md.update({"device_id": device_id, "day": day, "subset": "raw"})
        perday_records.append(md)

    # ---- Step 6: HDOP threshold subsets (in parallel) ----
    if "HDOP" in filt.columns:
        for thr in hdop_thresholds:
            sub = filt[filt["HDOP"] <= thr].copy()
            tag = f"HDOPle{thr}"
            m_sub = metrics_from_errors(sub["Error_m"].to_numpy(),
                                        sub["dx_m"].to_numpy(),
                                        sub["dy_m"].to_numpy())
            m_sub.update({"device_id": device_id, "subset": tag})
            overall_records.append(m_sub)

            for day, g in sub.groupby("day"):
                md = metrics_from_errors(g["Error_m"].to_numpy(), g["dx_m"].to_numpy(), g["dy_m"].to_numpy())
                md.update({"device_id": device_id, "day": day, "subset": tag})
                perday_records.append(md)

            # Plots per subset (optional; keep number moderate)
            if len(sub) > 0:
                plot_error_histogram(sub[["LAT","LON","Error_m"]], plots_dir / f"{device_id}_{tag}_hist.png",
                                     title=f"{device_id} {tag}: GPS Error Distribution")
                plot_scatter_vs_true(sub, true_lat, true_lon, plots_dir / f"{device_id}_{tag}_scatter_vs_true.png",
                                     title=f"{device_id} {tag}: GPS vs Mean Center")
                plot_hdop_vs_error(sub, plots_dir / f"{device_id}_{tag}_hdop_vs_error.png",
                                   title=f"{device_id} {tag}: HDOP vs Error")

    # ---- Step 7: Save error details + summary (your original outputs) ----
    # Use your filenames at per-device level
    filt.rename(columns={"Error_m":"Error (m)"}, inplace=True)
    filt.to_csv(out_dir / "gps_error_data.csv", index=False)

    summary = {
        "Mean Error (m)": round(filt["Error (m)"].mean(), 2),
        "Max Error (m)": round(filt["Error (m)"].max(), 2),
        "Standard Deviation (m)": round(filt["Error (m)"].std(ddof=1), 2)
    }
    pd.DataFrame([summary]).to_csv(out_dir / "gps_error_summary.csv", index=False)

    # ---- Step 8: Your two plots for RAW ----
    plot_error_histogram(filt[["LAT","LON","Error (m)"]].rename(columns={"Error (m)":"Error_m"}),
                         plots_dir / f"{device_id}_raw_hist.png",
                         title=f"{device_id} raw: GPS Error Distribution")
    plot_scatter_vs_true(filt.rename(columns={"Error (m)":"Error_m"}), true_lat, true_lon,
                         plots_dir / f"{device_id}_raw_scatter_vs_true.png",
                         title=f"{device_id} raw: GPS vs Mean Center")
    plot_hdop_vs_error(filt.rename(columns={"Error (m)":"Error_m"}),
                       plots_dir / f"{device_id}_raw_hdop_vs_error.png",
                       title=f"{device_id} raw: HDOP vs Error")
    plot_hdop_error_boxplot(filt.rename(columns={"Error (m)":"Error_m"}),
                            plots_dir / f"{device_id}_raw_hdop_vs_error_boxplot.png",
                            title=f"{device_id} raw: Error vs HDOP thresholds (boxplot)")

    # ---- Step 9: Export center + tabular summaries ----
    centers_df = pd.DataFrame([{"device_id": device_id, "lat_mean": true_lat, "lon_mean": true_lon}])
    centers_df.to_csv(out_dir / "device_centers.csv", index=False)

    overall_df = pd.DataFrame(overall_records)[[
        "device_id","subset","n_fixes","Mean_Error_m","Max_Error_m","Std_Error_m",
        "CEP50_m","CEP95_m","RMS_H_m","avg_r_m","2DRMS_m","std_dx_m","std_dy_m"
    ]].sort_values(["device_id","subset"])
    overall_df.to_csv(out_dir / "device_overall.csv", index=False)

    perday_df = pd.DataFrame(perday_records)[[
        "device_id","day","subset","n_fixes","Mean_Error_m","Max_Error_m","Std_Error_m",
        "CEP50_m","CEP95_m","RMS_H_m","avg_r_m","2DRMS_m","std_dx_m","std_dy_m"
    ]].sort_values(["device_id","day","subset"])
    perday_df.to_csv(out_dir / "device_perday.csv", index=False)

    print(f"[OK] Device {device_id}: outputs at {out_dir}")

# ----------------------------- Batch main -----------------------------
def main(root_dir: str = ROOT_DIR, out_root: str = OUT_ROOT):
    root = Path(root_dir)
    out_root = Path(out_root)
    ensure_dir(out_root)

    device_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not device_dirs:
        print(f"[WARN] No device folders under {root}")
        return

    for d in sorted(device_dirs):
        process_device_folder(d, out_root)

if __name__ == "__main__":
    main()
