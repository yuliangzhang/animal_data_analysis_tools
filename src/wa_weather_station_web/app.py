from __future__ import annotations

import os
import re
import smtplib
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    Response,
    url_for,
)

from wa_weather_station_tool.downloader import DownloadConfig, download_to_csv


# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = REPO_ROOT / "work_dir_web"
DOWNLOADS_DIR = WORK_DIR / "downloads"
DB_PATH = WORK_DIR / "requests.sqlite3"


@dataclass
class AppConfig:
    api_key: str
    base_url: str
    smtp_host: str
    smtp_port: int
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    smtp_sender: Optional[str] = None
    admin_user: Optional[str] = None
    admin_pass: Optional[str] = None


def _load_yaml_config() -> Dict[str, Any]:
    """Load YAML config from repo root `config.yml` if present.

    The expected structure:
    dpird_api_key: "..."
    base_url: "http://localhost:5000"  # public URL of this service
    smtp:
      host: "smtp.example.com"
      port: 587
      username: "user"
      password: "pass"
      use_tls: true
      sender: "noreply@example.com"
    """
    config_path = REPO_ROOT / "config.yml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_app_config() -> AppConfig:
    data = _load_yaml_config()
    api_key = data.get("dpird_api_key")
    base_url = data.get("base_url") or os.getenv("BASE_URL", "http://localhost:5000")
    smtp = data.get("smtp", {}) or {}
    admin = data.get("admin", {}) or {}

    if not api_key:
        raise RuntimeError(
            "Missing dpird_api_key in config.yml. Please add your API key."
        )

    return AppConfig(
        api_key=api_key,
        base_url=str(base_url).rstrip("/"),
        smtp_host=str(smtp.get("host", "")),
        smtp_port=int(smtp.get("port", 587)),
        smtp_username=smtp.get("username"),
        smtp_password=smtp.get("password"),
        smtp_use_tls=bool(smtp.get("use_tls", True)),
        smtp_sender=smtp.get("sender") or smtp.get("username"),
        admin_user=os.getenv("ADMIN_USER", admin.get("username") or "yuliang.zhang"),
        admin_pass=os.getenv("ADMIN_PASS", admin.get("password") or "zyl.123"),
    )


def _ensure_dirs() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _init_db() -> None:
    _ensure_dirs()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                station_code TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                utilization TEXT,
                filename TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT
            )
            """
        )
        conn.commit()


@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def _sanitize_filename_part(s: str) -> str:
    s = s.strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _compose_output_filename(email: str, station: str, start: str, end: str) -> str:
    parts = [
        _sanitize_filename_part(email),
        _sanitize_filename_part(station),
        _sanitize_filename_part(start),
        _sanitize_filename_part(end),
    ]
    return "_".join(p for p in parts if p) + ".csv"


def _send_email(cfg: AppConfig, to_email: str, subject: str, body: str) -> None:
    if not cfg.smtp_host:
        # Graceful noop if SMTP not configured; log to console
        print("[email] SMTP host not configured; skipping email send.")
        print("To:", to_email)
        print("Subject:", subject)
        print("Body:\n", body)
        return

    msg = EmailMessage()
    msg["From"] = cfg.smtp_sender or cfg.smtp_username or "noreply@example.com"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30) as server:
        if cfg.smtp_use_tls:
            server.starttls()
        if cfg.smtp_username and cfg.smtp_password:
            server.login(cfg.smtp_username, cfg.smtp_password)
        server.send_message(msg)


def _process_request_async(app_cfg: AppConfig, record_id: int) -> None:
    """Background worker: fetch data, update DB, and email user."""
    try:
        with db_conn() as conn:
            cur = conn.execute(
                "SELECT email, station_code, start_date, end_date FROM requests WHERE id=?",
                (record_id,),
            )
            row = cur.fetchone()
            if not row:
                return
            email, station, start_date, end_date = row

        # Build output path
        filename = _compose_output_filename(email, station, start_date, end_date)
        out_path = DOWNLOADS_DIR / filename

        # Run the downloader
        cfg = DownloadConfig(
            station_id=station,
            start_date_time=start_date,
            end_date_time=end_date,
            out_csv=out_path,
            api_key=app_cfg.api_key,
        )
        download_to_csv(cfg)

        # Update DB status
        now = datetime.utcnow().isoformat()
        with db_conn() as conn:
            conn.execute(
                "UPDATE requests SET status=?, filename=?, updated_at=? WHERE id=?",
                ("completed", filename, now, record_id),
            )
            conn.commit()

        # Email user with download link
        link = f"{app_cfg.base_url}/downloads/{filename}"
        subject = "Your WA DPIRD weather data is ready"
        body = (
            "Hello,\n\n"
            "Your weather data download has completed. You can download your file here:\n"
            f"{link}\n\n"
            "Best regards,\nWA Weather Data Service"
        )
        _send_email(app_cfg, email, subject, body)

    except Exception as e:  # noqa: BLE001
        now = datetime.utcnow().isoformat()
        with db_conn() as conn:
            conn.execute(
                "UPDATE requests SET status=?, updated_at=?, error=? WHERE id=?",
                ("failed", now, str(e), record_id),
            )
            conn.commit()


def _copy_station_image(static_dir: Path) -> None:
    """Copy provided station image into static dir at runtime if missing."""
    src_img = REPO_ROOT / "DPIRD_WA_Weather_Station.png"
    dst_img = static_dir / "DPIRD_WA_Weather_Station.png"
    try:
        if src_img.exists() and not dst_img.exists():
            static_dir.mkdir(parents=True, exist_ok=True)
            dst_img.write_bytes(src_img.read_bytes())
    except Exception:
        # Non-fatal; page will just show a missing image
        pass


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")

    # Ensure workspace and DB
    _ensure_dirs()
    _init_db()

    # Prepare static assets
    _copy_station_image(Path(app.static_folder))

    # Load config at startup and cache
    app.config["APP_CFG"] = _build_app_config()

    @app.route("/health")
    def health() -> str:
        return "ok"

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/submit", methods=["POST"])
    def submit():
        app_cfg: AppConfig = app.config["APP_CFG"]

        email = (request.form.get("email") or "").strip()
        station = (request.form.get("station") or "").strip()
        start_date_raw = (request.form.get("start") or "").strip()
        end_date_raw = (request.form.get("end") or "").strip()
        utilization = (request.form.get("utilization") or "").strip()

        # Basic validation
        if not email or "@" not in email:
            flash("Please provide a valid email.", "error")
            return redirect(url_for("index"))
        if not station:
            flash("Station code is required.", "error")
            return redirect(url_for("index"))
        if not start_date_raw or not end_date_raw:
            flash("Start and End dates are required.", "error")
            return redirect(url_for("index"))

        # Convert date-only to required format YYYY-MM-DDTHH:MM:SS (UTC day bounds)
        try:
            start_dt = datetime.strptime(start_date_raw, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date_raw, "%Y-%m-%d")
        except ValueError:
            flash("Invalid date format. Use calendar selector.", "error")
            return redirect(url_for("index"))

        if end_dt < start_dt:
            flash("End date cannot be before start date.", "error")
            return redirect(url_for("index"))

        # Enforce max 1-year range (inclusive)
        if (end_dt - start_dt) > timedelta(days=365):
            flash("Date range cannot exceed 1 year.", "error")
            return redirect(url_for("index"))

        start_str = start_dt.strftime("%Y-%m-%dT00:00:00")
        end_str = end_dt.strftime("%Y-%m-%dT23:59:59")

        # Insert request into DB with pending status
        now = datetime.utcnow().isoformat()
        with db_conn() as conn:
            cur = conn.execute(
                (
                    "INSERT INTO requests (email, station_code, start_date, end_date, "
                    "utilization, filename, status, created_at, updated_at, error) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
                (
                    email,
                    station,
                    start_str,
                    end_str,
                    utilization,
                    None,
                    "pending",
                    now,
                    now,
                    None,
                ),
            )
            request_id = cur.lastrowid
            conn.commit()

        # Start background thread to process the request
        t = threading.Thread(
            target=_process_request_async, args=(app.config["APP_CFG"], request_id), daemon=True
        )
        t.start()

        return render_template("submitted.html", email=email)

    @app.route("/downloads/<path:filename>")
    def download_file(filename: str):
        return send_from_directory(DOWNLOADS_DIR, filename, as_attachment=True)

    def _check_auth(auth: Optional[Any], cfg: AppConfig) -> bool:
        return (
            auth is not None
            and auth.type == "basic"
            and auth.username == (cfg.admin_user or "")
            and auth.password == (cfg.admin_pass or "")
        )

    def _auth_required_response() -> Response:
        return Response(
            "Authentication required",
            401,
            {"WWW-Authenticate": 'Basic realm="Admin Area"'},
        )

    @app.route("/admin")
    def admin():
        app_cfg: AppConfig = app.config["APP_CFG"]
        auth = request.authorization
        if not _check_auth(auth, app_cfg):
            return _auth_required_response()
        # Simple listing of latest requests; not authenticated for now.
        try:
            limit = int(request.args.get("limit", "200"))
        except ValueError:
            limit = 200
        rows: list[dict[str, Any]] = []
        with db_conn() as conn:
            cur = conn.execute(
                (
                    "SELECT id, email, station_code, start_date, end_date, utilization, "
                    "filename, status, created_at, updated_at, error "
                    "FROM requests ORDER BY created_at DESC LIMIT ?"
                ),
                (limit,),
            )
            cols = [c[0] for c in cur.description]
            for r in cur.fetchall():
                rows.append({k: v for k, v in zip(cols, r)})
        return render_template("admin.html", rows=rows)

    return app


if __name__ == "__main__":  # pragma: no cover
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
