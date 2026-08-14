"""SQLite-backed persistence with a conservative legacy JSON importer.

The application still treats a learner state as one JSON-compatible document,
but ownership and persistence are explicit: user -> books -> learner state.
Keeping the repository API narrow makes a future PostgreSQL implementation a
storage concern rather than an application-wide rewrite.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT_DIR / "state" / "langbuddy.sqlite3"
DEFAULT_LEGACY_STATE_DIR = ROOT_DIR / "state"
DEFAULT_USER_ID = "local-user"
USER_HEADER = "X-User-ID"

_USER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,63}")
_BOOK_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "langbuddy_user_id", default=DEFAULT_USER_ID
)


class MigrationConflictError(RuntimeError):
    """Raised when a legacy file changed after it was imported into SQLite."""


def validate_user_id(user_id: str) -> str:
    value = str(user_id or "").strip()
    if not _USER_ID_RE.fullmatch(value):
        raise ValueError("invalid user id")
    return value


def validate_book_id(book_id: str) -> str:
    value = str(book_id or "").strip()
    if not _BOOK_ID_RE.fullmatch(value):
        raise ValueError("invalid book id")
    return value


def current_user_id() -> str:
    return _current_user_id.get()


def set_current_user(user_id: str) -> contextvars.Token[str]:
    return _current_user_id.set(validate_user_id(user_id))


def reset_current_user(token: contextvars.Token[str]) -> None:
    _current_user_id.reset(token)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class MigrationSummary:
    books_imported: int = 0
    materials_imported: int = 0


class SQLiteStorage:
    def __init__(self, db_path: Path, legacy_state_dir: Optional[Path] = None):
        self.db_path = Path(db_path)
        self.legacy_state_dir = Path(legacy_state_dir) if legacy_state_dir else None
        self._initialized = False
        self._lock = threading.RLock()
        self.migration_summary = MigrationSummary()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS books (
                        user_id TEXT NOT NULL,
                        book_id TEXT NOT NULL,
                        source TEXT,
                        state_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, book_id),
                        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS material_stores (
                        user_id TEXT NOT NULL,
                        book_id TEXT NOT NULL,
                        store_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (user_id, book_id),
                        FOREIGN KEY (user_id, book_id)
                            REFERENCES books(user_id, book_id) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS legacy_imports (
                        source_path TEXT PRIMARY KEY,
                        checksum TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        book_id TEXT NOT NULL,
                        imported_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_books_user_updated
                        ON books(user_id, updated_at DESC);
                    """
                )
            if self.legacy_state_dir:
                self.migration_summary = self._migrate_legacy_json()
            self._initialized = True

    def _ensure_user(self, connection: sqlite3.Connection, user_id: str) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?)",
            (user_id, _utc_now()),
        )

    def _legacy_record(
        self, connection: sqlite3.Connection, path: Path
    ) -> Optional[sqlite3.Row]:
        return connection.execute(
            "SELECT checksum, user_id, book_id FROM legacy_imports WHERE source_path = ?",
            (str(path.resolve()),),
        ).fetchone()

    def _record_legacy_import(
        self,
        connection: sqlite3.Connection,
        path: Path,
        digest: str,
        book_id: str,
    ) -> None:
        connection.execute(
            """INSERT OR REPLACE INTO legacy_imports
               (source_path, checksum, user_id, book_id, imported_at)
               VALUES (?, ?, ?, ?, ?)""",
            (str(path.resolve()), digest, DEFAULT_USER_ID, book_id, _utc_now()),
        )

    def _migrate_legacy_json(self) -> MigrationSummary:
        state_dir = self.legacy_state_dir
        if not state_dir or not state_dir.exists():
            return MigrationSummary()
        imported_books = 0
        imported_materials = 0
        state_files = sorted(
            path for path in state_dir.glob("*.json") if path.is_file()
        )
        with self._connect() as connection:
            self._ensure_user(connection, DEFAULT_USER_ID)
            for path in state_files:
                raw = path.read_bytes()
                digest = _checksum(raw)
                previous = self._legacy_record(connection, path)
                if previous:
                    if previous["checksum"] != digest:
                        raise MigrationConflictError(
                            f"legacy learner state changed after SQLite import: {path}"
                        )
                    continue
                try:
                    state = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    raise MigrationConflictError(f"invalid legacy learner state: {path}") from exc
                if not isinstance(state, dict):
                    raise MigrationConflictError(f"invalid legacy learner state: {path}")
                book_id = validate_book_id(state.get("book_id") or path.stem)
                state["book_id"] = book_id
                existing = connection.execute(
                    "SELECT state_json FROM books WHERE user_id = ? AND book_id = ?",
                    (DEFAULT_USER_ID, book_id),
                ).fetchone()
                if existing and _canonical_json(json.loads(existing["state_json"])) != _canonical_json(state):
                    raise MigrationConflictError(
                        f"SQLite already contains different learner state for {book_id}"
                    )
                timestamp = datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat()
                if not existing:
                    connection.execute(
                        """INSERT INTO books
                           (user_id, book_id, source, state_json, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            DEFAULT_USER_ID,
                            book_id,
                            state.get("source"),
                            _canonical_json(state),
                            timestamp,
                            timestamp,
                        ),
                    )
                    imported_books += 1
                self._record_legacy_import(connection, path, digest, book_id)

            materials_dir = state_dir / "materials"
            if materials_dir.exists():
                for path in sorted(materials_dir.glob("*.json")):
                    raw = path.read_bytes()
                    digest = _checksum(raw)
                    previous = self._legacy_record(connection, path)
                    if previous:
                        if previous["checksum"] != digest:
                            raise MigrationConflictError(
                                f"legacy material store changed after SQLite import: {path}"
                            )
                        continue
                    try:
                        store = json.loads(raw.decode("utf-8"))
                    except Exception as exc:
                        raise MigrationConflictError(f"invalid legacy material store: {path}") from exc
                    if not isinstance(store, dict):
                        raise MigrationConflictError(f"invalid legacy material store: {path}")
                    book_id = validate_book_id(store.get("book_id") or path.stem)
                    owner = connection.execute(
                        "SELECT 1 FROM books WHERE user_id = ? AND book_id = ?",
                        (DEFAULT_USER_ID, book_id),
                    ).fetchone()
                    if not owner:
                        raise MigrationConflictError(
                            f"material store has no matching learner state: {path}"
                        )
                    existing = connection.execute(
                        """SELECT store_json FROM material_stores
                           WHERE user_id = ? AND book_id = ?""",
                        (DEFAULT_USER_ID, book_id),
                    ).fetchone()
                    if existing and _canonical_json(json.loads(existing["store_json"])) != _canonical_json(store):
                        raise MigrationConflictError(
                            f"SQLite already contains different materials for {book_id}"
                        )
                    if not existing:
                        connection.execute(
                            """INSERT INTO material_stores
                               (user_id, book_id, store_json, updated_at)
                               VALUES (?, ?, ?, ?)""",
                            (DEFAULT_USER_ID, book_id, _canonical_json(store), _utc_now()),
                        )
                        imported_materials += 1
                    self._record_legacy_import(connection, path, digest, book_id)
        return MigrationSummary(imported_books, imported_materials)

    def save_book_state(
        self, state: Mapping[str, Any], user_id: Optional[str] = None
    ) -> None:
        self.initialize()
        owner = validate_user_id(user_id or current_user_id())
        book_id = validate_book_id(str(state.get("book_id") or ""))
        payload = dict(state)
        payload["book_id"] = book_id
        now = _utc_now()
        with self._connect() as connection:
            self._ensure_user(connection, owner)
            connection.execute(
                """INSERT INTO books
                   (user_id, book_id, source, state_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, book_id) DO UPDATE SET
                     source = excluded.source,
                     state_json = excluded.state_json,
                     updated_at = excluded.updated_at""",
                (
                    owner,
                    book_id,
                    payload.get("source"),
                    _canonical_json(payload),
                    now,
                    now,
                ),
            )

    def load_book_state(
        self, book_id: str, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        self.initialize()
        owner = validate_user_id(user_id or current_user_id())
        safe_book_id = validate_book_id(book_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM books WHERE user_id = ? AND book_id = ?",
                (owner, safe_book_id),
            ).fetchone()
        return json.loads(row["state_json"]) if row else None

    def book_exists(self, book_id: str, user_id: Optional[str] = None) -> bool:
        return self.load_book_state(book_id, user_id) is not None

    def list_books(self, user_id: Optional[str] = None) -> list[Dict[str, Any]]:
        self.initialize()
        owner = validate_user_id(user_id or current_user_id())
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT book_id, source, state_json, created_at, updated_at
                   FROM books WHERE user_id = ? ORDER BY updated_at DESC""",
                (owner,),
            ).fetchall()
        items = []
        for row in rows:
            state = json.loads(row["state_json"])
            items.append({
                "bookId": row["book_id"],
                "source": row["source"],
                "group_count": len(state.get("groups") or {}),
                "created_at": row["created_at"],
            })
        return items

    def load_material_store(
        self, book_id: str, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        self.initialize()
        owner = validate_user_id(user_id or current_user_id())
        safe_book_id = validate_book_id(book_id)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT store_json FROM material_stores
                   WHERE user_id = ? AND book_id = ?""",
                (owner, safe_book_id),
            ).fetchone()
        return json.loads(row["store_json"]) if row else None

    def save_material_store(
        self,
        book_id: str,
        store: Mapping[str, Any],
        user_id: Optional[str] = None,
    ) -> None:
        self.initialize()
        owner = validate_user_id(user_id or current_user_id())
        safe_book_id = validate_book_id(book_id)
        if not self.book_exists(safe_book_id, owner):
            raise FileNotFoundError(f"book not found: {safe_book_id}")
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO material_stores(user_id, book_id, store_json, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, book_id) DO UPDATE SET
                     store_json = excluded.store_json,
                     updated_at = excluded.updated_at""",
                (owner, safe_book_id, _canonical_json(dict(store)), _utc_now()),
            )

    def delete_material_store(
        self, book_id: str, user_id: Optional[str] = None
    ) -> bool:
        self.initialize()
        owner = validate_user_id(user_id or current_user_id())
        safe_book_id = validate_book_id(book_id)
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM material_stores WHERE user_id = ? AND book_id = ?",
                (owner, safe_book_id),
            )
        return cursor.rowcount > 0


class BookStateRecord:
    """Small compatibility adapter for legacy call sites that used pathlib.Path."""

    def __init__(self, book_id: str, user_id: Optional[str] = None):
        self.book_id = validate_book_id(book_id)
        self.user_id = validate_user_id(user_id or current_user_id())

    def exists(self) -> bool:
        return get_storage().book_exists(self.book_id, self.user_id)

    def read_text(self, encoding: str = "utf-8") -> str:
        del encoding
        state = get_storage().load_book_state(self.book_id, self.user_id)
        if state is None:
            raise FileNotFoundError(self.book_id)
        return json.dumps(state, ensure_ascii=False, indent=2)

    def write_text(self, data: str, encoding: str = "utf-8") -> int:
        del encoding
        state = json.loads(data)
        if not isinstance(state, dict):
            raise ValueError("learner state must be a JSON object")
        state["book_id"] = self.book_id
        get_storage().save_book_state(state, self.user_id)
        return len(data)

    def stat(self) -> SimpleNamespace:
        return SimpleNamespace(st_mtime=datetime.now(timezone.utc).timestamp())


_storage: Optional[SQLiteStorage] = None
_storage_lock = threading.RLock()


def get_storage() -> SQLiteStorage:
    global _storage
    with _storage_lock:
        if _storage is None:
            db_path = Path(os.getenv("LANGBUDDY_DB_PATH", DEFAULT_DB_PATH))
            legacy_dir_raw = os.getenv("LANGBUDDY_LEGACY_STATE_DIR")
            legacy_dir = Path(legacy_dir_raw) if legacy_dir_raw else DEFAULT_LEGACY_STATE_DIR
            _storage = SQLiteStorage(db_path, legacy_dir)
        return _storage


def configure_storage(
    db_path: Path, legacy_state_dir: Optional[Path] = None
) -> SQLiteStorage:
    """Install an isolated repository, primarily for deterministic tests."""
    global _storage
    with _storage_lock:
        _storage = SQLiteStorage(db_path, legacy_state_dir)
        return _storage
