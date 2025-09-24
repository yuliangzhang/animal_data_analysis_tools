#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Markdown-based GPS accuracy reports (CN & EN), then convert to PDF.

The converter is self-contained (Matplotlib PDF backend) to avoid external
dependencies like pandoc/LaTeX. It parses a small Markdown subset:
- Headings: #, ##, ###
- Bullets:  - item
- Images:   ![alt](path)

Outputs:
  gps_accuracy_analysis/report/report_cn.md / .pdf
  gps_accuracy_analysis/report/report_en.md / .pdf
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, findSystemFonts

try:
    import requests  # only used to fetch CJK font if missing
except Exception:
    requests = None


ROOT = Path(__file__).resolve().parents[1] / "gps_accuracy_analysis"
RES_DIR = ROOT / "gps_analysis_res"
REPORT_DIR = ROOT / "report"
FONTS_DIR = REPORT_DIR / "fonts"


def ensure_dirs():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FONTS_DIR.mkdir(parents=True, exist_ok=True)


def find_or_fetch_cn_font() -> Optional[FontProperties]:
    candidates = [
        "NotoSansCJK", "SourceHanSans", "WenQuanYi", "SimHei", "Microsoft YaHei", "PingFang",
    ]
    for fpath in findSystemFonts() or []:
        lower = Path(fpath).name.lower()
        if any(k.lower() in lower for k in candidates):
            try:
                return FontProperties(fname=fpath)
            except Exception:
                continue
    if requests is None:
        return None
    url = (
        "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/"
        "SimplifiedChinese/SourceHanSansSC-Regular.otf"
    )
    dst = FONTS_DIR / "SourceHanSansSC-Regular.otf"
    if not dst.exists():
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dst.write_bytes(r.content)
        except Exception:
            return None
    try:
        return FontProperties(fname=str(dst))
    except Exception:
        return None


def load_device_metrics(device_id: str) -> Dict[str, Any]:
    ddir = RES_DIR / device_id
    overall_csv = ddir / "device_overall.csv"
    if not overall_csv.exists():
        raise FileNotFoundError(f"Missing metrics: {overall_csv}")
    df = pd.read_csv(overall_csv)

    def pick(subset: str) -> Dict[str, Any]:
        row = df[df["subset"] == subset]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {
            "n_fixes": int(r.get("n_fixes", 0)),
            "Mean_Error_m": float(r.get("Mean_Error_m", float("nan"))),
            "CEP50_m": float(r.get("CEP50_m", float("nan"))),
            "CEP95_m": float(r.get("CEP95_m", float("nan"))),
            "2DRMS_m": float(r.get("2DRMS_m", float("nan"))),
        }

    plots = {
        "hist": ddir / "plots" / f"{device_id}_raw_hist.png",
        "scatter": ddir / "plots" / f"{device_id}_raw_scatter_vs_true.png",
        "box": ddir / "plots" / f"{device_id}_raw_hdop_vs_error_boxplot.png",
    }
    return {"raw": pick("raw"), "HDOPle300": pick("HDOPle300"), "plots": plots}


def md_for_cn(m171: Dict[str, Any], m172: Dict[str, Any]) -> str:
    def mline(tag: Dict[str, Any], name: str) -> str:
        return (
            f"- {name}（n={tag.get('n_fixes',0)}）\n"
            f"  - 平均误差：{tag.get('Mean_Error_m', float('nan')):.1f} m\n"
            f"  - 一半点位在 {tag.get('CEP50_m', float('nan')):.1f} m 内（CEP50）\n"
            f"  - 95% 在 {tag.get('CEP95_m', float('nan')):.1f} m 内（CEP95）\n"
            f"  - 2DRMS：{tag.get('2DRMS_m', float('nan')):.1f} m"
        )

    md = [
        "# GPS设备精度测试报告（静止测试）",
        "",
        "结论：两台设备在静止测试中的水平定位精度均为米级；GPS171 明显优于 GPS172。",
        "",
        "## 核心指标（全部数据）",
        mline(m171["raw"], "GPS171"),
        "",
        mline(m172["raw"], "GPS172"),
        "",
        "## 良好信号（HDOP≤3.0）",
        f"- GPS171：平均误差 {m171['HDOPle300'].get('Mean_Error_m', float('nan')):.1f} m，95% 在 {m171['HDOPle300'].get('CEP95_m', float('nan')):.1f} m 内",
        f"- GPS172：平均误差 {m172['HDOPle300'].get('Mean_Error_m', float('nan')):.1f} m，95% 在 {m172['HDOPle300'].get('CEP95_m', float('nan')):.1f} m 内",
        "",
        "## 证据图（点击可查看原图）",
        "### GPS171",
        f"![GPS171 误差分布直方图](../gps_analysis_res/GPS171/plots/GPS171_raw_hist.png)",
        f"![GPS171 GPS点与均值位置](../gps_analysis_res/GPS171/plots/GPS171_raw_scatter_vs_true.png)",
        f"![GPS171 误差 vs HDOP（箱线）](../gps_analysis_res/GPS171/plots/GPS171_raw_hdop_vs_error_boxplot.png)",
        "",
        "### GPS172",
        f"![GPS172 误差分布直方图](../gps_analysis_res/GPS172/plots/GPS172_raw_hist.png)",
        f"![GPS172 GPS点与均值位置](../gps_analysis_res/GPS172/plots/GPS172_raw_scatter_vs_true.png)",
        f"![GPS172 误差 vs HDOP（箱线）](../gps_analysis_res/GPS172/plots/GPS172_raw_hdop_vs_error_boxplot.png)",
        "",
        "## 为什么 GPS171 更好（可优化要点）",
        "- 更好的卫星几何/信号质量（低 HDOP 占比更高）。",
        "- 安装与环境：尽量开阔天空、远离金属与墙面，减少遮挡与多路径反射。",
        "- 天线朝上固定，避免贴近身体或地面；保持设备稳定以减少噪声。",
        "- 采集前预热 1–2 分钟；遇到高 HDOP 时可延迟采集或换位置。",
        "- 后处理建议：过滤 HDOP>3.0（本项目使用 HDOP×100 阈值），去除零经纬度点。",
    ]
    return "\n".join(md)


def md_for_en(m171: Dict[str, Any], m172: Dict[str, Any]) -> str:
    def mline(tag: Dict[str, Any], name: str) -> str:
        return (
            f"- {name} (n={tag.get('n_fixes',0)})\n"
            f"  - Mean: {tag.get('Mean_Error_m', float('nan')):.1f} m\n"
            f"  - 50% within {tag.get('CEP50_m', float('nan')):.1f} m (CEP50)\n"
            f"  - 95% within {tag.get('CEP95_m', float('nan')):.1f} m (CEP95)\n"
            f"  - 2DRMS: {tag.get('2DRMS_m', float('nan')):.1f} m"
        )

    md = [
        "# GPS Accuracy Test Report (Static)",
        "",
        "Bottom line: both devices are meter-level; GPS171 outperforms GPS172.",
        "",
        "## Headline Metrics (All Data)",
        mline(m171["raw"], "GPS171"),
        "",
        mline(m172["raw"], "GPS172"),
        "",
        "## Good Signal (HDOP ≤ 3.0)",
        f"- GPS171: Mean {m171['HDOPle300'].get('Mean_Error_m', float('nan')):.1f} m; 95% within {m171['HDOPle300'].get('CEP95_m', float('nan')):.1f} m",
        f"- GPS172: Mean {m172['HDOPle300'].get('Mean_Error_m', float('nan')):.1f} m; 95% within {m172['HDOPle300'].get('CEP95_m', float('nan')):.1f} m",
        "",
        "## Visual Evidence",
        "### GPS171",
        f"![GPS171 Error Histogram](../gps_analysis_res/GPS171/plots/GPS171_raw_hist.png)",
        f"![GPS171 Points vs Mean Center](../gps_analysis_res/GPS171/plots/GPS171_raw_scatter_vs_true.png)",
        f"![GPS171 Error vs HDOP (Boxplot)](../gps_analysis_res/GPS171/plots/GPS171_raw_hdop_vs_error_boxplot.png)",
        "",
        "### GPS172",
        f"![GPS172 Error Histogram](../gps_analysis_res/GPS172/plots/GPS172_raw_hist.png)",
        f"![GPS172 Points vs Mean Center](../gps_analysis_res/GPS172/plots/GPS172_raw_scatter_vs_true.png)",
        f"![GPS172 Error vs HDOP (Boxplot)](../gps_analysis_res/GPS172/plots/GPS172_raw_hdop_vs_error_boxplot.png)",
        "",
        "## Why GPS171 Performs Better (Actionable)",
        "- Stronger satellite geometry/signal quality (more low-HDOP fixes).",
        "- Mounting & environment: open sky, away from metal/walls, less obstruction/multipath.",
        "- Keep antenna facing sky and stable; avoid close contact with body/ground.",
        "- Warm up for 1–2 minutes; when HDOP is high, pause or relocate.",
        "- Post-process with HDOP filtering (≤3.0; this project uses HDOP×100 thresholds) and drop zero lat/lon fixes.",
    ]
    return "\n".join(md)


IMG_RE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<path>[^)]+)\)")


def wrap_text(s: str, limit: int = 84) -> List[str]:
    # simple wrapper for readability; avoid breaking URLs/paths
    out: List[str] = []
    for line in s.splitlines():
        if len(line) <= limit:
            out.append(line)
        else:
            # naive wrap at spaces; if none, hard wrap
            buf = line
            while len(buf) > limit:
                cut = buf.rfind(" ", 0, limit)
                if cut == -1:
                    out.append(buf[:limit])
                    buf = buf[limit:]
                else:
                    out.append(buf[:cut])
                    buf = buf[cut+1:]
            if buf:
                out.append(buf)
    return out


def convert_md_to_pdf(md_path: Path, pdf_path: Path, cn_font: Optional[FontProperties] = None):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    # Expand lines to keep them short (better layout)
    lines = [l for L in [wrap_text(l) for l in lines] for l in L]

    with PdfPages(str(pdf_path)) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        y = 0.95
        left = 0.08
        right = 0.92
        width = right - left

        def new_page():
            nonlocal fig, ax, y
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            y = 0.95
            return fig, ax

        for raw in lines:
            if y < 0.10:
                fig, ax = new_page()
            if not raw.strip():
                y -= 0.02
                continue

            # Images
            m = IMG_RE.search(raw)
            if m:
                img_rel = m.group('path')
                img_path = (md_path.parent / img_rel).resolve()
                if img_path.exists():
                    # draw image block
                    try:
                        img = plt.imread(str(img_path))
                        # target width (in figure coords)
                        target_w = width
                        # Convert figure coords to axes using add_axes
                        ih, iw = img.shape[0], img.shape[1]
                        aspect = ih / iw if iw else 1.0
                        target_h = target_w * aspect * 0.6  # scale for readability
                        # ensure space; if not, new page
                        if y - target_h < 0.10:
                            fig, ax = new_page()
                        ax_img = fig.add_axes([left, y - target_h, target_w, target_h])
                        ax_img.imshow(img)
                        ax_img.axis('off')
                        y -= target_h + 0.02
                        continue
                    except Exception:
                        pass  # fallback to printing the raw line

            # Headings & bullets
            font_kw = {}
            if cn_font is not None:
                font_kw = dict(fontproperties=cn_font)

            if raw.startswith('# '):
                ax.text(left, y, raw[2:].strip(), fontsize=20, weight='bold', **font_kw)
                y -= 0.06
            elif raw.startswith('## '):
                ax.text(left, y, raw[3:].strip(), fontsize=16, weight='bold', **font_kw)
                y -= 0.05
            elif raw.startswith('### '):
                ax.text(left, y, raw[4:].strip(), fontsize=14, weight='bold', **font_kw)
                y -= 0.04
            elif raw.startswith('- '):
                ax.text(left + 0.01, y, '• ' + raw[2:].strip(), fontsize=12, **font_kw)
                y -= 0.03
            elif raw.startswith('  - '):
                ax.text(left + 0.04, y, '· ' + raw[4:].strip(), fontsize=12, **font_kw)
                y -= 0.03
            else:
                ax.text(left, y, raw.strip(), fontsize=12, **font_kw)
                y -= 0.03

        pdf.savefig(fig)
        plt.close(fig)


def build_reports_md_pdf():
    ensure_dirs()
    cn_font = find_or_fetch_cn_font()

    m171 = load_device_metrics("GPS171")
    m172 = load_device_metrics("GPS172")

    # Markdown files
    md_cn = md_for_cn(m171, m172)
    md_en = md_for_en(m171, m172)

    md_cn_path = REPORT_DIR / "report_cn.md"
    md_en_path = REPORT_DIR / "report_en.md"
    md_cn_path.write_text(md_cn, encoding="utf-8")
    md_en_path.write_text(md_en, encoding="utf-8")

    # Convert MD -> PDF (self-contained)
    convert_md_to_pdf(md_cn_path, REPORT_DIR / "report_cn.pdf", cn_font)
    convert_md_to_pdf(md_en_path, REPORT_DIR / "report_en.pdf", None)

    print(f"[OK] Markdown and PDFs saved to: {REPORT_DIR}")


if __name__ == "__main__":
    build_reports_md_pdf()

