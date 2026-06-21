"""Structured logging to ~/.viyon/logs/ (SQLite + JSON). Never logs secrets.

Every VIYON command is recorded in a SQLite ``commands`` table and mirrored as a
human-readable JSON file at ``<base_dir>/<date>/<id>.json``. Records pass through
:meth:`VLogger.scrub` first so API keys never reach disk.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_DIR = Path("~/.viyon/logs").expanduser()

# Columns that hold JSON-encoded values in the database.
_JSON_COLUMNS = ("parsed_intent", "agents", "steps")

# Every column on the commands table, in insert order.
_COLUMNS = (
    "ts",
    "raw_input",
    "parsed_intent",
    "agents",
    "steps",
    "result",
    "duration_ms",
    "confirmed",
    "status",
    "error",
)

_REDACTED = "***REDACTED***"

# Keys named in .env — redact "<NAME> = <value>" / "<NAME>: <value>" assignments.
_ENV_KEY_RE = re.compile(
    r"(?i)\b(ANTHROPIC_API_KEY|ELEVENLABS_API_KEY|PICOVOICE_ACCESS_KEY|BRAVE_API_KEY)\b"
    r"(\s*[=:]\s*)(\S+)"
)

# Raw secret-shaped tokens that should never appear anywhere.
_TOKEN_RES = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),  # Anthropic
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),    # generic "sk-" keys
    re.compile(r"\bxi-[A-Za-z0-9]{16,}\b"),    # ElevenLabs-style
)

# Dict keys whose *value* is a secret regardless of content.
_SECRET_KEY_NAME_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?key|secret|token|password|passwd|pwd|bearer|authorization)"
)


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class VLogger:
    """Writes structured, secret-scrubbed command records to ~/.viyon/logs/.

    Args:
        base_dir: Directory for the SQLite db and JSON mirrors. Defaults to
            ``~/.viyon/logs``; tests may point this at a temporary directory.
    """

    def __init__(self, base_dir: Path | str = DEFAULT_LOG_DIR) -> None:
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "viyon.db"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        """Create the ``commands`` table if it does not yet exist."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS commands (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts            TEXT    NOT NULL,
                    raw_input     TEXT,
                    parsed_intent TEXT,
                    agents        TEXT,
                    steps         TEXT,
                    result        TEXT,
                    duration_ms   INTEGER,
                    confirmed     INTEGER,
                    status        TEXT,
                    error         TEXT
                )
                """
            )

    # -- secret scrubbing ---------------------------------------------------

    @classmethod
    def scrub(cls, obj: Any) -> Any:
        """Return a deep copy of ``obj`` with anything secret-shaped redacted.

        Handles nested dicts and lists. Dict values whose *key* looks like a
        credential name are redacted wholesale; all strings are additionally
        scanned for token- and ``KEY=value``-shaped secrets.
        """
        if isinstance(obj, dict):
            scrubbed: dict[Any, Any] = {}
            for key, value in obj.items():
                if isinstance(key, str) and _SECRET_KEY_NAME_RE.search(key):
                    scrubbed[key] = _REDACTED
                else:
                    scrubbed[key] = cls.scrub(value)
            return scrubbed
        if isinstance(obj, (list, tuple)):
            return [cls.scrub(item) for item in obj]
        if isinstance(obj, str):
            return cls._scrub_str(obj)
        return obj

    @staticmethod
    def _scrub_str(text: str) -> str:
        """Redact secret-shaped substrings inside a single string."""
        text = _ENV_KEY_RE.sub(rf"\1\2{_REDACTED}", text)
        for pattern in _TOKEN_RES:
            text = pattern.sub(_REDACTED, text)
        return text

    # -- writes -------------------------------------------------------------

    def log_command(self, record: dict) -> int:
        """Insert a command record and return its new row id.

        ``record`` may supply any of the table columns; ``ts`` defaults to now
        and ``status`` to ``"ok"``. The record is scrubbed before persisting and
        mirrored to ``<base_dir>/<date>/<id>.json``.
        """
        record = self.scrub(dict(record))
        record.setdefault("ts", _now_iso())
        record.setdefault("status", "ok")

        values = [self._encode(col, record.get(col)) for col in _COLUMNS]
        placeholders = ", ".join("?" for _ in _COLUMNS)
        sql = f"INSERT INTO commands ({', '.join(_COLUMNS)}) VALUES ({placeholders})"
        with self._lock, self._conn:
            cursor = self._conn.execute(sql, values)
            row_id = int(cursor.lastrowid)

        self._write_mirror(row_id)
        return row_id

    def update_command(self, id: int, **fields: Any) -> None:
        """Update columns of an existing command and refresh its JSON mirror.

        Raises:
            ValueError: if ``fields`` names a column that does not exist, or if
                no row with ``id`` exists.
        """
        unknown = set(fields) - set(_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown command column(s): {sorted(unknown)}")
        if not fields:
            return

        fields = self.scrub(dict(fields))
        assignments = ", ".join(f"{col} = ?" for col in fields)
        values = [self._encode(col, val) for col, val in fields.items()]
        values.append(id)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"UPDATE commands SET {assignments} WHERE id = ?", values
            )
            if cursor.rowcount == 0:
                raise ValueError(f"No command with id={id}")

        self._write_mirror(id)

    def _write_mirror(self, id: int) -> None:
        """Write/refresh the JSON mirror file for the given command id."""
        record = self.get_by_id(id)
        if record is None:
            return
        date = str(record.get("ts", ""))[:10] or datetime.now(timezone.utc).date().isoformat()
        day_dir = self.base_dir / date
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / f"{id}.json").write_text(
            json.dumps(record, indent=2, default=str, ensure_ascii=False)
        )

    # -- reads --------------------------------------------------------------

    def get_by_id(self, id: int) -> dict | None:
        """Return a single command as a dict, or None if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM commands WHERE id = ?", (id,)
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def get_recent(self, n: int = 20) -> list[dict]:
        """Return the ``n`` most recent commands, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM commands ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search(self, query: str) -> list[dict]:
        """Return commands whose text columns contain ``query`` (newest first)."""
        like = f"%{query}%"
        cols = ("raw_input", "parsed_intent", "agents", "steps", "result", "error")
        where = " OR ".join(f"{col} LIKE ?" for col in cols)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM commands WHERE {where} ORDER BY id DESC",
                tuple(like for _ in cols),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _encode(column: str, value: Any) -> Any:
        """Encode a Python value for storage in the given column."""
        if value is None:
            return None
        if column in _JSON_COLUMNS:
            return json.dumps(value, default=str, ensure_ascii=False)
        if column == "confirmed":
            return int(bool(value))
        return value

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """Convert a DB row into a dict, decoding JSON columns and booleans."""
        record = dict(row)
        for col in _JSON_COLUMNS:
            if record.get(col) is not None:
                try:
                    record[col] = json.loads(record[col])
                except (json.JSONDecodeError, TypeError):
                    pass
        if record.get("confirmed") is not None:
            record["confirmed"] = bool(record["confirmed"])
        return record

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()


# Module-level singleton used across VIYON.
log = VLogger()
