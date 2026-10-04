"""
SQLite implementation of :class:`~mra_lib.storage.records.RegimeStore`.

Uses the standard-library :mod:`sqlite3` module only.

Thread-safety:
    For a file database every operation opens its own short-lived connection
    (closed when the operation ends). Connections are never shared between
    threads, so the store can be used from any number of worker threads, and
    from several processes at once (e.g. API workers plus a scanner). The
    database runs in WAL mode: readers never block the writer and vice versa,
    and concurrent writers wait up to :data:`BUSY_TIMEOUT_SECONDS` for the
    write lock. The per-call ``connect`` costs well under a millisecond, which
    is negligible next to an analysis.

    An in-memory database (``":memory:"``, for tests) only exists for the
    lifetime of its connection, so it uses a single connection opened with
    ``check_same_thread=False`` and serializes every operation with a lock.

Timestamps are stored as fixed-width ISO 8601 text in UTC without an offset
(``YYYY-MM-DDTHH:MM:SS.ffffff``), so string order is chronological order.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mra_lib.errors import StorageError, StorageInputError

from .records import (
    DEFAULT_HISTORY_LIMIT,
    MAX_HISTORY_LIMIT,
    RegimeRecord,
    normalize_symbol,
    to_naive_utc,
)

logger = logging.getLogger(__name__)

#: Environment variable overriding the database path.
DB_PATH_ENV = "MRA_DB_PATH"
#: Default database path (``~`` is expanded at resolution time).
DEFAULT_DB_PATH = "~/.mra/regimes.db"
#: Special path for a private, non-persistent database.
MEMORY_PATH = ":memory:"
#: Current schema version, stored in ``PRAGMA user_version``.
SCHEMA_VERSION = 1
#: Seconds a connection waits for a lock held by another writer.
BUSY_TIMEOUT_SECONDS = 5.0

_TABLE = "regime_records"
_COLUMNS = (
    "symbol",
    "timeframe",
    "bar_time",
    "recorded_at",
    "regime",
    "confidence",
    "persistence",
    "transition_probability",
    "recommended_strategy",
    "close",
    "provider",
)

# Version 1 schema. The unique index is both the upsert key and the lookup index
# for latest() and history(symbol, timeframe, ...).
_SCHEMA_V1 = (
    f"""
    CREATE TABLE IF NOT EXISTS {_TABLE} (
        id INTEGER PRIMARY KEY,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        bar_time TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        regime TEXT NOT NULL,
        confidence REAL NOT NULL,
        persistence REAL NOT NULL,
        transition_probability REAL NOT NULL,
        recommended_strategy TEXT NOT NULL,
        close REAL,
        provider TEXT NOT NULL
    )
    """,
    f"""
    CREATE UNIQUE INDEX IF NOT EXISTS ux_{_TABLE}_symbol_timeframe_bar
        ON {_TABLE} (symbol, timeframe, bar_time)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS ix_{_TABLE}_symbol_bar
        ON {_TABLE} (symbol, bar_time)
    """,
)

_UPSERT = (
    f"INSERT INTO {_TABLE} ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _COLUMNS)}) "
    "ON CONFLICT (symbol, timeframe, bar_time) DO UPDATE SET "
    + ", ".join(
        f"{c} = excluded.{c}" for c in _COLUMNS if c not in {"symbol", "timeframe", "bar_time"}
    )
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE}"


def _format_time(value: datetime) -> str:
    """Fixed-width UTC text for a datetime (naive values are taken as UTC)."""
    return to_naive_utc(value).isoformat(timespec="microseconds")


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _row_to_record(row: sqlite3.Row) -> RegimeRecord:
    return RegimeRecord(
        symbol=row["symbol"],
        timeframe=row["timeframe"],
        bar_time=_parse_time(row["bar_time"]),
        recorded_at=_parse_time(row["recorded_at"]).replace(tzinfo=UTC),
        regime=row["regime"],
        confidence=row["confidence"],
        persistence=row["persistence"],
        transition_probability=row["transition_probability"],
        recommended_strategy=row["recommended_strategy"],
        close=row["close"],
        provider=row["provider"],
    )


def _record_params(record: RegimeRecord) -> tuple[Any, ...]:
    return (
        record.symbol,
        record.timeframe,
        _format_time(record.bar_time),
        _format_time(record.recorded_at),
        record.regime,
        record.confidence,
        record.persistence,
        record.transition_probability,
        record.recommended_strategy,
        record.close,
        record.provider,
    )


def resolve_db_path(path: str | os.PathLike[str] | None = None) -> str:
    """Resolve the database path.

    Args:
        path: Explicit path. When ``None``, ``$MRA_DB_PATH`` is used if set and
            non-empty, otherwise :data:`DEFAULT_DB_PATH`.

    Returns:
        ``":memory:"`` or an absolute filesystem path with ``~`` expanded.
    """
    raw = os.fspath(path) if path is not None else os.getenv(DB_PATH_ENV, "").strip()
    if not raw:
        raw = DEFAULT_DB_PATH
    if raw == MEMORY_PATH:
        return MEMORY_PATH
    return str(Path(raw).expanduser().absolute())


class SQLiteRegimeStore:
    """Regime history in a SQLite database.

    The schema is created (or migrated) on first use; nothing touches the
    filesystem until then. See the module docstring for the thread-safety model.

    Args:
        path: Database file, or ``":memory:"`` for a private in-memory database.
            ``None`` resolves via :func:`resolve_db_path` (``$MRA_DB_PATH`` or
            ``~/.mra/regimes.db``). The parent directory is created with mode
            ``0o700`` and a new database file with mode ``0o600``.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = resolve_db_path(path)
        self._init_lock = threading.Lock()
        self._initialized = False
        self._memory_conn: sqlite3.Connection | None = None
        self._memory_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path!r})"

    @property
    def is_memory(self) -> bool:
        """Whether this store is a private in-memory database."""
        return self.path == MEMORY_PATH

    # ── connection handling ──────────────────────────────────────────────

    def _open(self) -> sqlite3.Connection:
        if self.is_memory:
            conn = sqlite3.connect(MEMORY_PATH, check_same_thread=False, isolation_level=None)
        else:
            self._prepare_file()
            conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _prepare_file(self) -> None:
        """Create the parent directory (0o700) and an empty database file (0o600)."""
        db_file = Path(self.path)
        db_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not db_file.exists():
            flags = os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
            os.close(os.open(db_file, flags, 0o600))

    def _initialize(self, conn: sqlite3.Connection) -> None:
        """Enable WAL and create or migrate the schema (once per store instance).

        An up-to-date database is only read; the write lock is taken only when
        the schema has to be created or migrated.
        """
        if not self.is_memory:
            mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            if mode != "wal":
                mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
                if mode != "wal":
                    logger.warning("SQLite WAL mode unavailable for %s (got %s)", self.path, mode)
        if self._user_version(conn) == SCHEMA_VERSION:
            return
        # BEGIN IMMEDIATE takes the write lock, so concurrent processes cannot
        # both create/migrate the schema; re-read the version under the lock.
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = self._user_version(conn)
            if version > SCHEMA_VERSION:
                raise StorageError(
                    f"Regime database schema version {version} is newer than "
                    f"supported version {SCHEMA_VERSION}"
                )
            if version < 1:
                for statement in _SCHEMA_V1:
                    conn.execute(statement)
            # Future migrations: `if version < 2: ...` here, in order.
            if version != SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION:d}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _user_version(conn: sqlite3.Connection) -> int:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a ready connection; wrap ``sqlite3`` / OS errors in :class:`StorageError`."""
        try:
            if self.is_memory:
                with self._memory_lock:
                    if self._memory_conn is None:
                        self._memory_conn = self._open()
                    conn = self._memory_conn
                    self._ensure_initialized(conn)
                    yield conn
                return
            conn = self._open()
            try:
                self._ensure_initialized(conn)
                yield conn
            finally:
                conn.close()
        except StorageError:
            raise
        except (sqlite3.Error, OSError) as e:
            # Re-check the schema next time (e.g. the file was deleted or replaced)
            self._initialized = False
            logger.warning("Regime store %s operation failed: %s", self.path, e)
            raise StorageError("Regime store operation failed") from e

    def _ensure_initialized(self, conn: sqlite3.Connection) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if not self._initialized:
                self._initialize(conn)
                self._initialized = True

    def close(self) -> None:
        """Close the in-memory connection (discarding its data); no-op for files."""
        with self._memory_lock:
            if self._memory_conn is not None:
                self._memory_conn.close()
                self._memory_conn = None
                self._initialized = False

    # ── public API ───────────────────────────────────────────────────────

    def schema_version(self) -> int:
        """Return the database's ``PRAGMA user_version`` (creating the schema if needed)."""
        with self._connection() as conn:
            return self._user_version(conn)

    def save(self, record: RegimeRecord) -> None:
        """Insert ``record``, replacing any record for the same (symbol, timeframe, bar_time).

        Args:
            record: The record to store.

        Raises:
            StorageInputError: If ``record`` is not a :class:`RegimeRecord`.
            StorageError: If the database cannot be written.
        """
        if not isinstance(record, RegimeRecord):
            raise StorageInputError("record must be a RegimeRecord")
        params = _record_params(record)
        with self._connection() as conn:
            conn.execute(_UPSERT, params)

    def latest(self, symbol: str, timeframe: str) -> RegimeRecord | None:
        """Return the record with the newest ``bar_time`` for (symbol, timeframe).

        Args:
            symbol: Ticker symbol (case-insensitive).
            timeframe: Timeframe, matched exactly.

        Returns:
            The newest record, or ``None`` if there is none.

        Raises:
            StorageError: If the database cannot be read.
        """
        sym = normalize_symbol(symbol)
        with self._connection() as conn:
            row = conn.execute(
                f"{_SELECT} WHERE symbol = ? AND timeframe = ? ORDER BY bar_time DESC LIMIT 1",
                (sym, timeframe),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def history(
        self,
        symbol: str,
        timeframe: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> list[RegimeRecord]:
        """Return records for ``symbol``, newest ``bar_time`` first.

        Args:
            symbol: Ticker symbol (case-insensitive).
            timeframe: Only this timeframe; ``None`` for all timeframes.
            since: Inclusive lower bound on ``bar_time`` (naive = UTC).
            until: Inclusive upper bound on ``bar_time`` (naive = UTC).
            limit: Maximum number of records; values above
                :data:`MAX_HISTORY_LIMIT` are capped to it.

        Returns:
            Matching records, newest first (an empty list for unknown symbols).

        Raises:
            StorageInputError: If ``limit`` is not a positive integer.
            StorageError: If the database cannot be read.
        """
        sym = normalize_symbol(symbol)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise StorageInputError("limit must be a positive integer")
        limit = min(limit, MAX_HISTORY_LIMIT)

        clauses = ["symbol = ?"]
        params: list[Any] = [sym]
        if timeframe is not None:
            clauses.append("timeframe = ?")
            params.append(timeframe)
        if since is not None:
            clauses.append("bar_time >= ?")
            params.append(_format_time(since))
        if until is not None:
            clauses.append("bar_time <= ?")
            params.append(_format_time(until))
        params.append(limit)

        query = (
            f"{_SELECT} WHERE {' AND '.join(clauses)} ORDER BY bar_time DESC, timeframe ASC LIMIT ?"
        )
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def symbols(self) -> list[str]:
        """Return every symbol with at least one record, sorted.

        Raises:
            StorageError: If the database cannot be read.
        """
        with self._connection() as conn:
            rows = conn.execute(f"SELECT DISTINCT symbol FROM {_TABLE} ORDER BY symbol").fetchall()
        return [row["symbol"] for row in rows]


_default_stores: dict[str, SQLiteRegimeStore] = {}
_default_stores_lock = threading.Lock()


def default_store() -> SQLiteRegimeStore:
    """Return the store for the configured database (``$MRA_DB_PATH`` or ``~/.mra/regimes.db``).

    The path is resolved on every call, so a changed ``$MRA_DB_PATH`` takes
    effect immediately. File stores are cached per path (so the schema check
    runs once per process); ``":memory:"`` always gets a fresh, private store.
    Creating the store does not touch the filesystem.
    """
    path = resolve_db_path()
    if path == MEMORY_PATH:
        logger.warning(
            "%s=:memory: gives every default_store() call a private, empty database; "
            "nothing is persisted or shared",
            DB_PATH_ENV,
        )
        return SQLiteRegimeStore(MEMORY_PATH)
    with _default_stores_lock:
        store = _default_stores.get(path)
        if store is None:
            store = _default_stores[path] = SQLiteRegimeStore(path)
        return store
