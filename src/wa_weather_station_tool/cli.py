from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from . import __version__
from .downloader import DownloadConfig, download_to_csv


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI.

    This minimal version only parses common downloader options
    and prints them back. Real downloading will be implemented later.
    """
    parser = argparse.ArgumentParser(
        prog="wa_dpird_weather_downloader",
        description=(
            "Download WA DPIRD weather station data (minimal CLI scaffold)."
        ),
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
        help="Start date (YYYY-MM-DD). Optional for minimal version.",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=False,
        default=None,
        help="End date (YYYY-MM-DD). Optional for minimal version.",
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
        required=False,
        default=None,
        help=(
            "DPIRD API key. If not provided, reads from env var "
            "DPIRD_API_KEY or .env if available."
        ),
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

    # Resolve API key from args, env or dotenv
    api_key = args.api_key
    if api_key is None:
        # try dotenv if available
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass
        import os

        api_key = os.getenv("DPIRD_API_KEY")

    if not api_key:
        parser.error(
            "API key is required. Provide --api-key or set DPIRD_API_KEY in environment/.env."
        )

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
