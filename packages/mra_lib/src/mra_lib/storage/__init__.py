"""
Regime history storage.

Persist :class:`RegimeRecord` objects (one regime classification per
symbol, timeframe and bar) and query them later::

    from mra_lib.storage import RegimeRecord, default_store

    store = default_store()  # $MRA_DB_PATH or ~/.mra/regimes.db
    analysis = analyzer.analyze_current_regime("1D")
    store.save(RegimeRecord.from_analysis(analysis, analyzer, "1D"))
    store.latest("SPY", "1D")
    store.history("SPY", timeframe="1D", limit=20)

:class:`RegimeStore` is the protocol; :class:`SQLiteRegimeStore` is the
standard-library ``sqlite3`` implementation.
"""

from .records import (
    DEFAULT_HISTORY_LIMIT,
    MAX_HISTORY_LIMIT,
    RegimeRecord,
    RegimeStore,
)
from .sqlite_store import (
    DB_PATH_ENV,
    DEFAULT_DB_PATH,
    SCHEMA_VERSION,
    SQLiteRegimeStore,
    default_store,
    resolve_db_path,
)

__all__ = [
    "DB_PATH_ENV",
    "DEFAULT_DB_PATH",
    "DEFAULT_HISTORY_LIMIT",
    "MAX_HISTORY_LIMIT",
    "SCHEMA_VERSION",
    "RegimeRecord",
    "RegimeStore",
    "SQLiteRegimeStore",
    "default_store",
    "resolve_db_path",
]
