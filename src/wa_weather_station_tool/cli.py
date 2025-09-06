from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from . import __version__
from .downloader import DownloadConfig, download_to_csv


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI.

    Downloader for WA DPIRD weather station data.
    """
    parser = argparse.ArgumentParser(
        prog="wa_dpird_weather_downloader",
        description=("Download WA DPIRD weather station data."),
    )

    parser.add_argument(
        "--station",
        type=str,
        required=True,
        help="Station ID/code to download (e.g., '009225').",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=False,
        default=None,
        help="Start date-time (YYYY-MM-DDTHH:MM:SS, UTC)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=False,
        default=None,
        help="End date-time (YYYY-MM-DDTHH:MM:SS, UTC)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        default=Path("weather_data.csv"),
        help="Output CSV file path (default: weather_data.csv).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="DPIRD API key (required).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Page size for API pagination (default: 200).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"wa_dpird_weather_downloader {__version__}",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the minimal CLI.

    For now, it validates arguments and writes a small placeholder CSV
    to confirm the tool is wired up correctly.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # API key is required via CLI flag
    api_key = args.api_key

    cfg = DownloadConfig(
        station_id=args.station,
        start_date_time=args.start or "",
        end_date_time=args.end or "",
        out_csv=args.out,
        api_key=api_key,
        limit=int(args.limit),
    )

    download_to_csv(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
