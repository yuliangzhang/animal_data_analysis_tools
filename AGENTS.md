# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Python packages and reusable modules.
- `tests/`: Unit/integration tests mirroring `src/` layout.
- `scripts/`: One-off or CLI utilities (training/eval/data prep).
- `datasets/`: Read-only raw data; keep large assets out of Git.
- `work_dir*/`: Experiment outputs, logs, checkpoints (gitignored).
- Notebooks: Keep exploratory `.ipynb` at root or `notebooks/`; promote stable code into `src/`.

## Build, Test, and Development Commands
- Environment (choose one):
  - Conda: `conda env update -f environment.yml && conda activate <env>`
  - venv: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Lint/format: `ruff check .`, `black .`, `isort .`
- Run tests: `pytest -q` (filter: `pytest -k <pattern>`)
- Run scripts/modules: `python scripts/<name>.py` or `python -m <package>.<module>`

## Coding Style & Naming Conventions
- Python: 4-space indentation; format with Black (line length 88).
- Naming: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Prefer type hints and concise, informative docstrings (Google/NumPy style).
- Keep functions small and pure; avoid notebook-only logic in production modules.

## Testing Guidelines
- Framework: `pytest` with tests under `tests/` named `test_*.py`.
- Coverage: aim ≥80% for changed code; add tests for bugs before fixing.
- Use fixtures/mocks for data-heavy logic; avoid large files in tests.
- Quick run: `pytest --maxfail=1 --disable-warnings -q`

## Commit & Pull Request Guidelines
- Commits: Prefer Conventional Commits (e.g., `feat:`, `fix:`, `docs:`); present tense, concise subject, details in body.
- PRs: clear description and rationale, linked issues, screenshots for UI/plots, and checklist: tests updated, docs touched, lint/format pass.
- Keep changes focused; separate functional changes from formatting.

## Security & Configuration Tips
- Do not commit secrets (`.pem`, API keys), large media, or checkpoints. Use `.env` (load via `dotenv`) and ensure `.gitignore` covers `work_dir*/`, `*.ipynb_checkpoints`, `*.pt`, `*.pkl`, `*.mp4`.
- Parameterize paths and credentials via config or environment variables.

## Agent-Specific Notes
- Apply these guidelines across this directory tree.
- Make minimal, targeted patches; avoid unrelated refactors.
- Preserve structure and naming; update docs/tests when behavior changes.

