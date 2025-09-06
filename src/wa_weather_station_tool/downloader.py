from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests


API_BASE = "https://api.agric.wa.gov.au/v2/weather/stations/{station_id}/data"


@dataclass
class DownloadConfig:
    station_id: str
    start_date_time: str
    end_date_time: str
    out_csv: Path
    api_key: str
    limit: int = 200
    sort: str = "-dateTime"  # newest first, consistent with reference code
    sleep_between: float = 0.8  # polite delay between pages
    max_retries: int = 6  # for 429/5xx


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-GB,en;q=0.9,en-US;q=0.8,en-AU;q=0.7",
        "Connection": "keep-alive",
        "Origin": "https://weather.agric.wa.gov.au",
        "Referer": "https://weather.agric.wa.gov.au/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
        "api_key": api_key,
    }


def _build_url(cfg: DownloadConfig, offset: int) -> str:
    return (
        f"{API_BASE.format(station_id=cfg.station_id)}?"
        f"startDateTime={cfg.start_date_time}&endDateTime={cfg.end_date_time}"
        f"&offset={offset}&limit={cfg.limit}&sort={cfg.sort}"
    )


def _request_with_retries(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    max_retries: int,
) -> Dict[str, Any]:
    backoff = 1.0
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=30)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                # Rate-limited or server error: backoff and retry
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else backoff
                time.sleep(delay)
                backoff = min(backoff * 2, 30)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_exc = e
            # Network error: backoff and retry
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
    # All attempts failed
    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected retry loop termination")


def _parse_records(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    metadata = payload.get("metadata", {})
    collection_meta = metadata.get("collection", {})
    total_count = int(collection_meta.get("count", 0))

    records = payload.get("collection", [])
    parsed: List[Dict[str, Any]] = []
    for record in records:
        try:
            dt = datetime.strptime(record["dateTime"], "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            # Fallback: leave as original string if parsing fails
            dt = record.get("dateTime")

        wind = record.get("wind", []) or []
        wind0 = wind[0] if wind else {}
        wind_avg = (wind0 or {}).get("avg", {})
        wind_dir = (wind_avg or {}).get("direction", {})

        parsed.append(
            {
                "dateTime": dt,
                "airTemperature": record.get("airTemperature"),
                "relativeHumidity": record.get("relativeHumidity"),
                "soilTemperature": record.get("soilTemperature"),
                "solarIrradiance": record.get("solarIrradiance"),
                "rainfall": record.get("rainfall"),
                "dewPoint": record.get("dewPoint"),
                "deltaT": record.get("deltaT"),
                "wetBulb": record.get("wetBulb"),
                "batteryVoltage": record.get("batteryVoltage"),
                "wind_speed": wind_avg.get("speed"),
                "wind_direction": wind_dir.get("compassPoint"),
                "wind_degrees": wind_dir.get("degrees"),
            }
        )
    return parsed, total_count


def _count_existing_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8") as f:
        # subtract header if present
        rows = sum(1 for _ in f)
    return max(0, rows - 1)


def _write_rows(csv_path: Path, rows: Iterable[Dict[str, Any]], write_header: bool) -> int:
    import csv

    fieldnames = [
        "dateTime",
        "airTemperature",
        "relativeHumidity",
        "soilTemperature",
        "solarIrradiance",
        "rainfall",
        "dewPoint",
        "deltaT",
        "wetBulb",
        "batteryVoltage",
        "wind_speed",
        "wind_direction",
        "wind_degrees",
    ]

    count = 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            # Ensure datetime serialization
            if isinstance(row.get("dateTime"), datetime):
                row = dict(row)
                row["dateTime"] = row["dateTime"].strftime("%Y-%m-%dT%H:%M:%SZ")
            writer.writerow(row)
            count += 1
    return count


def _ckpt_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".ckpt.json")


def _save_ckpt(csv_path: Path, *, offset: int, total: int) -> None:
    ckpt = {"offset": offset, "total": total, "updated_at": time.time()}
    _ckpt_path(csv_path).write_text(json.dumps(ckpt, indent=2), encoding="utf-8")


def _load_ckpt(csv_path: Path) -> Tuple[int, int] | None:
    p = _ckpt_path(csv_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return int(data.get("offset", 0)), int(data.get("total", 0))
    except Exception:
        return None


def download_to_csv(cfg: DownloadConfig) -> None:
    headers = _build_headers(cfg.api_key)
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Determine resume offset
    offset = 0
    existing_rows = _count_existing_rows(cfg.out_csv)
    ckpt = _load_ckpt(cfg.out_csv)
    if ckpt:
        offset = max(offset, ckpt[0])
    offset = max(offset, existing_rows)

    write_header = not cfg.out_csv.exists()

    with requests.Session() as session:
        total = None
        while True:
            url = _build_url(cfg, offset)
            payload = _request_with_retries(
                session, url, headers=headers, max_retries=cfg.max_retries
            )
            rows, total_count = _parse_records(payload)
            total = total or total_count

            if not rows:
                break

            wrote = _write_rows(cfg.out_csv, rows, write_header)
            write_header = False
            offset += wrote
            _save_ckpt(cfg.out_csv, offset=offset, total=total_count)

            print(
                f"Fetched {offset}/{total_count} rows for station {cfg.station_id}..."
            )

            if offset >= total_count:
                break

            # polite sleep between requests
            time.sleep(cfg.sleep_between)

    print(f"Done. CSV saved at: {cfg.out_csv.resolve()}")

