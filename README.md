# animal_data_analysis_tools
Utility tools for animal experiments data analysis.

## WA DPIRD Weather Downloader
Command-line tool to fetch weather data from WA DPIRD with robust retries and resumable CSV output.

### Quick Start (PyPI)
- Install: `pip install wa-weather-station-tool`
- Show help: `wa_dpird_weather_downloader --help`
- Example:
  - `wa_dpird_weather_downloader --station WN --start 2024-03-19T00:00:00 --end 2024-04-19T00:00:00 --out datasets/WN_2024-03-19_to_2025-04-19.csv --api-key YOUR_API_KEY`

### Requirements
- Python 3.10+
- A valid DPIRD API key (pass via `--api-key`)

### Usage
Basic syntax:

`wa_dpird_weather_downloader --station <STATION_ID> --start <YYYY-MM-DDTHH:MM:SS> --end <YYYY-MM-DDTHH:MM:SS> --out <OUTPUT.csv> --api-key <YOUR_API_KEY>`

Examples:
- Download a range to CSV:
  `wa_dpird_weather_downloader --station 009225 --start 2024-01-01T00:00:00 --end 2024-01-31T23:59:59 --out datasets/009225_2024-01.csv --api-key YOUR_API_KEY`
- Download using station code `WN`:
  `wa_dpird_weather_downloader --station WN --station WN --start 2024-03-19T00:00:00 --end 2024-04-19T00:00:00 --out datasets/WN_2024-03-19_to_2025-04-19.csv --api-key YOUR_API_KEY`

### CLI Flags
- `--station`: Station ID/code (e.g., `WN`, `009225`).
- `--start`: Start date-time `YYYY-MM-DDTHH:MM:SS` (GMT).
- `--end`: End date-time `YYYY-MM-DDTHH:MM:SS` (GMT).
- `--out`: Output CSV path (e.g., `datasets/WN_2024-03-19_to_2025-04-19.csv`).
- `--api-key`: DPIRD API key (required). Can get from DPIRD official website (https://www.dpird.wa.gov.au/forms/dpird-api-registration/)
- `--limit`: Page size for API pagination (default 200).

### Features
- Retries with exponential backoff for HTTP 429/rate limiting and 5xx.
- Resumable downloads: appends page-by-page to CSV and writes a checkpoint file `<out>.ckpt.json`.
- Streamed writing to keep memory use low.

### Output Format (CSV columns)
- `dateTime`, `airTemperature`, `relativeHumidity`, `soilTemperature`, `solarIrradiance`,
  `rainfall`, `dewPoint`, `deltaT`, `wetBulb`, `batteryVoltage`, `wind_speed`,
  `wind_direction`, `wind_degrees`

### Resuming an Interrupted Download
- Re-run the same command with the same `--out` path; the tool resumes from the last page.
- It uses the row count and a checkpoint file `<out>.ckpt.json` to continue safely.

### Troubleshooting
- Rate limiting (429): The tool automatically retries with backoff. Reduce load with `--limit 100` if needed.
- Empty CSV: Ensure your `--start`/`--end` are in GMT and within the station’s data range.
- API key errors: Confirm the `--api-key` is correct and active.

### Contact / Support
- Email: yuliang.zhang@research.uwa.edu.au

When contacting support, please include:
- The full CLI command you ran (without exposing your API key).
- Error message/output and a brief description of the issue.
- Your OS, Python version, and package version (`wa-weather-station-tool --version`).

Or, if you are not familiar with Python/Coding, I can provide the downloading service.

### Local Install (alternative to PyPI)
- Download the repository: `git clone https://github.com/yuliangzhang/animal_data_analysis_tools.git`
- Change to the root directory: `cd animal_data_analysis_tools`
- From source root: `pip install -e .`
- Run CLI: `wa_dpird_weather_downloader --help`

<!-- Maintainer notes (hidden from typical users)
Build: python -m build
Upload to PyPI: twine upload dist/* (user: __token__)
-->
 
## Web Downloader (No Coding Required)
This repo includes a simple web app to request WA DPIRD weather data via a browser. It stores requests, downloads data in the background using the CLI downloader, and emails a download link when ready.

Locations:
- `src/wa_weather_station_web/`: Flask app, templates, routes
- `work_dir_web/`: SQLite DB and downloaded CSVs (gitignored)
- `config.yml`: App settings (see `config.yml.example`)

Run locally:
1. Create env and install deps
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Configure
   - Copy `config.yml.example` to `config.yml` and set `dpird_api_key` and SMTP settings.
   - Optionally export `BASE_URL` if not using `http://localhost:5000`.
3. Start
   - `python scripts/run_web.py`
4. Use
   - Open `http://localhost:5000` and submit the form.

Notes:
- Date range is limited to 1 year. The backend converts to `YYYY-MM-DDTHH:MM:SS`.
- Files are served at `/downloads/<filename>` from `work_dir_web/downloads/`.
- If SMTP is not configured, email is skipped and logged to console.

