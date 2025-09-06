# animal_data_analysis_tools
Develop the utility tools for animal experiments data analysis for the researchers.

## WA DPIRD Weather Downloader
- Command: `wa_dpird_weather_downloader --help`
- Example: `wa_dpird_weather_downloader --station WN --start 2024-03-19T00:00:00 --end 2025-08-01T00:00:00 --out datasets/WN_2024-03-19_to_2025-08-01.csv`

The CLI downloads real data from the WA DPIRD API with retries and resumable CSV output.

### Install locally for development
- Editable install: `pip install -e .`
- Run CLI after install: `wa_dpird_weather_downloader --help`

### Usage
- Set API key via env or `.env`:
  - `.env` example: `DPIRD_API_KEY=YOUR_API_KEY`
  - Or export: `export DPIRD_API_KEY=YOUR_API_KEY`
- Run:
  - `wa_dpird_weather_downloader --station WN --start 2024-03-19T00:00:00 --end 2025-08-01T00:00:00 --out datasets/WN_2024-03-19_to_2025-08-01.csv`

### Flags
- `--station`: station ID/code (e.g., `WN`, `009225`)
- `--start`, `--end`: date-time `YYYY-MM-DDTHH:MM:SS`
- `--out`: output CSV path
- `--api-key`: override env; otherwise reads `DPIRD_API_KEY` (loads `.env` if present)
- `--limit`: page size (default 200)

### Features
- Retries with exponential backoff for 429 and 5xx
- Resumable downloads: appends pages to CSV and maintains a sidecar checkpoint `<out>.ckpt.json`
- Page-by-page writing to keep memory usage low

### Build & publish to PyPI (summary)
1) Build: `python -m build` (install `build` first)
2) Upload: `twine upload dist/*` (use PyPI API token `__token__`)
See below for detailed steps.

### Detailed PyPI publish steps
1. Create PyPI account and API token (scoped to the project)
2. Install tooling: `pip install build twine`
3. Bump version in `pyproject.toml` under `[project] version`
4. Build artifacts: `python -m build` (creates `dist/*.tar.gz` & `dist/*.whl`)
5. Upload to TestPyPI (optional): `twine upload -r testpypi dist/*`
   - Then install to test: `pip install -i https://test.pypi.org/simple/ wa-weather-station-tool`
6. Upload to PyPI: `twine upload dist/*`
   - Username: `__token__`, Password: your API token
7. Verify install: `pip install wa-weather-station-tool && wa_dpird_weather_downloader --help`
