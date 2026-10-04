"""Tests for the regime history store (``mra_lib.storage``)."""

import contextlib
import sqlite3
import stat
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.enums import MarketRegime, TradingStrategy
from mra_lib.errors import MRAError, StorageError, StorageInputError
from mra_lib.storage import (
    DB_PATH_ENV,
    MAX_HISTORY_LIMIT,
    SCHEMA_VERSION,
    RegimeRecord,
    RegimeStore,
    SQLiteRegimeStore,
    default_store,
    resolve_db_path,
)

BASE = datetime(2026, 1, 5)


def make_record(
    symbol: str = "SPY",
    timeframe: str = "1D",
    bar_time: datetime = BASE,
    regime: str = "Bull Trending",
    **overrides,
) -> RegimeRecord:
    values = {
        "symbol": symbol,
        "timeframe": timeframe,
        "bar_time": bar_time,
        "regime": regime,
        "confidence": 0.8,
        "persistence": 0.6,
        "transition_probability": 0.9,
        "recommended_strategy": "Trend Following",
        "provider": "mock",
        "close": 101.5,
        "recorded_at": datetime(2026, 1, 5, 21, 0, tzinfo=UTC),
    }
    values.update(overrides)
    return RegimeRecord(**values)


@pytest.fixture
def store(tmp_path: Path) -> SQLiteRegimeStore:
    return SQLiteRegimeStore(tmp_path / "db" / "regimes.db")


@pytest.fixture
def memory_store():
    s = SQLiteRegimeStore(":memory:")
    yield s
    s.close()


class TestRegimeRecord:
    def test_record_normalizes_symbol_and_times(self):
        aware_bar = datetime(2026, 1, 5, 10, 0, tzinfo=timezone(timedelta(hours=-5)))
        rec = make_record(symbol=" spy ", bar_time=aware_bar, recorded_at=datetime(2026, 1, 5))
        assert rec.symbol == "SPY"
        assert rec.bar_time == datetime(2026, 1, 5, 15, 0)
        assert rec.bar_time.tzinfo is None
        assert rec.recorded_at == datetime(2026, 1, 5, tzinfo=UTC)

    def test_record_default_recorded_at_is_utc_now(self):
        rec = make_record(recorded_at=datetime.now(UTC))
        values = {k: getattr(rec, k) for k in rec.__dataclass_fields__ if k != "recorded_at"}
        fresh = RegimeRecord(**values)
        assert fresh.recorded_at.tzinfo is UTC
        assert abs(datetime.now(UTC) - fresh.recorded_at) < timedelta(seconds=5)

    def test_record_is_frozen(self):
        rec = make_record()
        with pytest.raises(AttributeError):
            rec.symbol = "QQQ"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"symbol": " "},
            {"symbol": "-SPY"},
            {"symbol": "A" * 16},
            {"bar_time": datetime(1, 1, 1, tzinfo=timezone(timedelta(hours=5)))},
            {"timeframe": ""},
            {"regime": ""},
            {"provider": ""},
            {"confidence": float("nan")},
            {"persistence": "high"},
            {"bar_time": "2026-01-05"},
            {"bar_time": pd.NaT},
            {"close": "x"},
        ],
    )
    def test_record_rejects_invalid_fields(self, overrides):
        with pytest.raises(StorageInputError):
            make_record(**overrides)

    def test_record_non_finite_close_becomes_none(self):
        assert make_record(close=float("inf")).close is None

    def test_storage_errors_are_mra_errors(self):
        assert issubclass(StorageError, MRAError)
        assert issubclass(StorageInputError, StorageError)
        assert issubclass(StorageInputError, ValueError)

    def test_record_from_analysis(self):
        idx = pd.DatetimeIndex([datetime(2026, 1, 2), datetime(2026, 1, 5)])
        df = pd.DataFrame({"Close": [100.0, 102.25]}, index=idx)
        analyzer = SimpleNamespace(
            symbol="qqq", data={"1D": df}, provider=SimpleNamespace(provider_name="mock")
        )
        analysis = RegimeAnalysis(
            current_regime=MarketRegime.HIGH_VOLATILITY,
            hmm_state=2,
            transition_probability=0.3,
            regime_persistence=0.4,
            recommended_strategy=TradingStrategy.VOLATILITY_TRADING,
            position_sizing_multiplier=0.5,
            risk_level="HIGH",
            arbitrage_opportunities=[],
            statistical_signals=[],
            key_levels={},
            regime_confidence=0.7,
        )
        rec = RegimeRecord.from_analysis(analysis, analyzer, "1D")  # type: ignore[arg-type]
        assert rec.symbol == "QQQ"
        assert rec.timeframe == "1D"
        assert rec.bar_time == datetime(2026, 1, 5)
        assert rec.regime == "High Volatility"
        assert rec.recommended_strategy == "Volatility Trading"
        assert rec.confidence == pytest.approx(0.7)
        assert rec.persistence == pytest.approx(0.4)
        assert rec.transition_probability == pytest.approx(0.3)
        assert rec.close == pytest.approx(102.25)
        assert rec.provider == "mock"
        assert rec.recorded_at.tzinfo is UTC

        with pytest.raises(StorageInputError):
            RegimeRecord.from_analysis(analysis, analyzer, "1H")  # type: ignore[arg-type]

    def test_record_from_real_analyzer(self):
        from mra_lib import MarketRegimeAnalyzer

        analyzer = MarketRegimeAnalyzer("SPY", periods={"1D": "2y"}, provider_flag="mock")
        analysis = analyzer.analyze_current_regime("1D")
        rec = RegimeRecord.from_analysis(analysis, analyzer, "1D")
        assert rec.bar_time == analyzer.data["1D"].index[-1].to_pydatetime()
        assert rec.regime == analysis.current_regime.value
        assert rec.provider == "mock"


class TestSQLiteRegimeStore:
    def test_store_satisfies_protocol(self, memory_store):
        assert isinstance(memory_store, RegimeStore)

    def test_store_round_trip(self, store):
        rec = make_record()
        store.save(rec)
        assert store.latest("spy", "1D") == rec
        assert store.history("SPY") == [rec]

    def test_store_round_trip_none_close_and_microseconds(self, memory_store):
        rec = make_record(close=None, bar_time=datetime(2026, 1, 5, 9, 30, 0, 123456))
        memory_store.save(rec)
        assert memory_store.latest("SPY", "1D") == rec

    def test_store_upserts_same_bar(self, store):
        store.save(make_record(regime="Bull Trending", confidence=0.5))
        store.save(make_record(regime="Bear Trending", confidence=0.9))
        records = store.history("SPY")
        assert len(records) == 1
        assert records[0].regime == "Bear Trending"
        assert records[0].confidence == pytest.approx(0.9)

    def test_store_upsert_key_matches_aware_and_naive_bar_time(self, memory_store):
        memory_store.save(make_record(bar_time=datetime(2026, 1, 5, 15, 0)))
        memory_store.save(make_record(bar_time=datetime(2026, 1, 5, 15, 0, tzinfo=UTC)))
        assert len(memory_store.history("SPY")) == 1

    def test_store_latest(self, memory_store):
        assert memory_store.latest("SPY", "1D") is None
        for days in (0, 2, 1):
            memory_store.save(make_record(bar_time=BASE + timedelta(days=days)))
        memory_store.save(make_record(timeframe="1H", bar_time=BASE + timedelta(days=5)))
        latest = memory_store.latest("SPY", "1D")
        assert latest is not None
        assert latest.bar_time == BASE + timedelta(days=2)
        assert memory_store.latest("QQQ", "1D") is None

    def test_store_history_order_and_filters(self, memory_store):
        for days in range(5):
            memory_store.save(make_record(bar_time=BASE + timedelta(days=days)))
            memory_store.save(make_record(timeframe="1H", bar_time=BASE + timedelta(days=days)))
        memory_store.save(make_record(symbol="QQQ"))

        all_tf = memory_store.history("SPY")
        assert len(all_tf) == 10
        times = [r.bar_time for r in all_tf]
        assert times == sorted(times, reverse=True)

        daily = memory_store.history("SPY", timeframe="1D")
        assert [r.bar_time for r in daily] == [BASE + timedelta(days=d) for d in (4, 3, 2, 1, 0)]

        window = memory_store.history(
            "SPY",
            timeframe="1D",
            since=BASE + timedelta(days=1),
            until=BASE + timedelta(days=3),
        )
        assert [r.bar_time for r in window] == [BASE + timedelta(days=d) for d in (3, 2, 1)]

        # Aware bounds are converted to UTC
        aware_since = (
            (BASE + timedelta(days=3)).replace(tzinfo=UTC).astimezone(timezone(timedelta(hours=2)))
        )
        assert len(memory_store.history("SPY", timeframe="1D", since=aware_since)) == 2

        assert len(memory_store.history("SPY", limit=3)) == 3
        assert memory_store.history("NOPE") == []

    def test_store_history_limit_cap(self, memory_store):
        for minutes in range(MAX_HISTORY_LIMIT + 5):
            memory_store.save(
                make_record(timeframe="15m", bar_time=BASE + timedelta(minutes=minutes))
            )
        assert len(memory_store.history("SPY", limit=MAX_HISTORY_LIMIT + 100)) == MAX_HISTORY_LIMIT

    @pytest.mark.parametrize("limit", [0, -1, True, 2.5])
    def test_store_history_rejects_bad_limit(self, memory_store, limit):
        with pytest.raises(StorageInputError):
            memory_store.history("SPY", limit=limit)

    def test_store_symbols(self, memory_store):
        assert memory_store.symbols() == []
        for sym in ("qqq", "SPY", "IWM", "SPY"):
            memory_store.save(make_record(symbol=sym))
        assert memory_store.symbols() == ["IWM", "QQQ", "SPY"]

    def test_store_rejects_non_record(self, memory_store):
        with pytest.raises(StorageInputError):
            memory_store.save({"symbol": "SPY"})  # type: ignore[arg-type]

    def test_store_schema_version_wal_and_indexes(self, store):
        assert store.schema_version() == SCHEMA_VERSION
        conn = sqlite3.connect(store.path)
        try:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            indexes = {row[1]: row[2] for row in conn.execute("PRAGMA index_list(regime_records)")}
            key = [i for i, unique in indexes.items() if unique]
            assert key
            cols = [row[2] for row in conn.execute(f"PRAGMA index_info({key[0]})")]
            assert cols == ["symbol", "timeframe", "bar_time"]
        finally:
            conn.close()

    def test_store_reopens_existing_database(self, store):
        store.save(make_record())
        again = SQLiteRegimeStore(store.path)
        assert again.latest("SPY", "1D") == make_record()

    def test_store_refuses_newer_schema(self, store):
        store.schema_version()
        conn = sqlite3.connect(store.path)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        conn.close()
        with pytest.raises(StorageError, match="newer"):
            SQLiteRegimeStore(store.path).symbols()

    def test_store_recovers_after_database_is_deleted(self, store):
        store.save(make_record())
        for suffix in ("", "-wal", "-shm"):
            Path(store.path + suffix).unlink(missing_ok=True)
        # The first call after the swap may fail; the schema is then re-checked
        with contextlib.suppress(StorageError):
            store.symbols()
        store.save(make_record(symbol="QQQ"))
        assert store.symbols() == ["QQQ"]

    def test_store_history_rejects_out_of_range_bound(self, memory_store):
        with pytest.raises(StorageInputError):
            memory_store.history(
                "SPY", since=datetime(1, 1, 1, tzinfo=timezone(timedelta(hours=5)))
            )

    def test_store_file_permissions(self, store):
        store.save(make_record())
        db = Path(store.path)
        assert stat.S_IMODE(db.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(db.stat().st_mode) == 0o600

    def test_store_wraps_sqlite_errors(self, tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        bad = SQLiteRegimeStore(blocker / "regimes.db")
        with pytest.raises(StorageError) as excinfo:
            bad.symbols()
        assert not isinstance(excinfo.value, StorageInputError)

    def test_store_corrupt_file_raises_storage_error(self, tmp_path):
        db = tmp_path / "corrupt.db"
        db.write_bytes(b"this is not a sqlite database" * 100)
        with pytest.raises(StorageError):
            SQLiteRegimeStore(db).history("SPY")

    def test_store_concurrent_writes_from_threads(self, store):
        n_threads, per_thread = 8, 25
        errors: list[BaseException] = []
        barrier = threading.Barrier(n_threads)

        def worker(i: int) -> None:
            try:
                barrier.wait()
                for j in range(per_thread):
                    store.save(
                        make_record(
                            symbol=f"S{i}",
                            timeframe="15m",
                            bar_time=BASE + timedelta(minutes=j),
                        )
                    )
                    store.latest(f"S{i}", "15m")
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert store.symbols() == sorted(f"S{i}" for i in range(n_threads))
        for i in range(n_threads):
            assert len(store.history(f"S{i}", limit=MAX_HISTORY_LIMIT)) == per_thread

    def test_memory_store_shared_across_threads(self, memory_store):
        def worker(i: int) -> None:
            memory_store.save(make_record(symbol=f"M{i}"))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert memory_store.symbols() == ["M0", "M1", "M2", "M3"]


class TestPathResolution:
    def test_default_path_from_env(self, tmp_path, monkeypatch):
        target = tmp_path / "custom" / "history.db"
        monkeypatch.setenv(DB_PATH_ENV, str(target))
        store = default_store()
        assert store.path == str(target)
        assert not target.parent.exists()  # nothing touched until first use
        store.save(make_record())
        assert target.exists()
        assert default_store() is store  # cached per path
        assert default_store().latest("SPY", "1D") == make_record()

    def test_default_path_without_env_uses_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv(DB_PATH_ENV, raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_db_path() == str(tmp_path / ".mra" / "regimes.db")
        monkeypatch.setenv(DB_PATH_ENV, "  ")
        assert resolve_db_path() == str(tmp_path / ".mra" / "regimes.db")

    def test_env_path_expands_user(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(DB_PATH_ENV, "~/x/r.db")
        assert resolve_db_path() == str(tmp_path / "x" / "r.db")

    def test_memory_path(self, monkeypatch, caplog):
        monkeypatch.setenv(DB_PATH_ENV, ":memory:")
        a, b = default_store(), default_store()
        assert "nothing is persisted" in caplog.text
        assert a.is_memory
        assert a is not b
        a.save(make_record())
        assert b.symbols() == []
        a.close()
