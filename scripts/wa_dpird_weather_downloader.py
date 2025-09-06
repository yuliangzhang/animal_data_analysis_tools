#!/usr/bin/env python
"""Convenience wrapper to run the minimal CLI.

Placed under scripts/ so it works without installing the package.
It appends the repository's `source/` directory to `sys.path` to
make the `wa_weather_station_tool` importable.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_source_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> int:
    _ensure_source_on_path()
    from wa_weather_station_tool.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
