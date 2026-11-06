"""Local-only SQLite inspection metadata storage; image bytes are never stored."""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

APP_NAME = "CastingInspectionAI"


def get_app_data_dir() -> Path:
    override = os.environ.get("CASTING_INSPECTION_APP_DATA")
    if override:
        root = Path(override).expanduser()
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / APP_NAME
    elif os.environ.get("XDG_DATA_HOME"):
        root = Path(os.environ["XDG_DATA_HOME"]).expanduser() / APP_NAME
    else:
        root = Path.home() / ".local" / "share" / APP_NAME
    for folder in (root, root / "database"):
        folder.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            try:
                folder.chmod(0o700)
            except OSError:
                pass
    return root


APP_DATA_DIR = get_app_data_dir()
DB_PATH = APP_DATA_DIR / "database" / "inspection_history.db"


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(str(DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _ensure_schema() -> None:
    with _connect() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_name TEXT NOT NULL,
            result TEXT NOT NULL,
            defect_type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.0
        )""")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_inspections_result ON inspections(result)")
        connection.commit()
    if os.name != "nt":
        try:
            DB_PATH.chmod(0o600)
        except OSError:
            pass


def _safe_image_name(image_name: str) -> str:
    name = str(image_name).replace(chr(92), "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return (name or "uploaded-image")[:255]


def add_inspection(image_name: str, result: str, defect_type: str, confidence: float) -> int:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with _connect() as connection:
        cursor = connection.execute(
            "INSERT INTO inspections(timestamp,image_name,result,defect_type,confidence) VALUES(?,?,?,?,?)",
            (timestamp, _safe_image_name(image_name), result, defect_type, float(confidence)),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_recent_inspections(limit: int = 15) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT timestamp,image_name,result,defect_type,confidence FROM inspections ORDER BY id DESC LIMIT ?",
            (max(1, min(int(limit), 500)),),
        ).fetchall()
    return [dict(row) for row in rows]


def get_last_30_minutes() -> list[dict]:
    cutoff = (datetime.now().astimezone() - timedelta(minutes=30)).isoformat(timespec="seconds")
    with _connect() as connection:
        rows = connection.execute(
            "SELECT timestamp,image_name,result,defect_type,confidence FROM inspections WHERE timestamp >= ? ORDER BY id DESC",
            (cutoff,),
        ).fetchall()
    return [dict(row) for row in rows]


def clear_inspections() -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM inspections")
        connection.commit()


_ensure_schema()
