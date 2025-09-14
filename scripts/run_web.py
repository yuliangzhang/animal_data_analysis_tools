from __future__ import annotations

"""Run the WA DPIRD Weather web service locally.

Usage:
  python scripts/run_web.py
"""

import os

import sys
from pathlib import Path

# Ensure local src/ is on path for dev runs without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wa_weather_station_web import create_app


def main() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)


if __name__ == "__main__":
    main()
