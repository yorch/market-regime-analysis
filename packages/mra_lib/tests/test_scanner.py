"""Tests for Scanner.scan_once / Scanner.run (fake analyzers, tmp_path store, no network)."""

import logging
import signal
from datetime import UTC, datetime, timedelta

import pytest

from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.enums import MarketRegime
from mra_lib.config.regime_tables import REGIME_STRATEGIES
from mra_lib.errors import InvalidParametersError, NotifierError, StorageError
from mra_lib.scanner import (
    AlertPolicy,
    ChangeKind,
    LogNotifier,
    RegimeChangeEvent,
    Scanner,
    SuppressReason,
    WebhookNotifier,
    format_scan_report,
    format_scan_summary,
)
from mra_lib.scheduling import StopReason
from mra_lib.storage import RegimeRecord, SQLiteRegimeStore

BULL = MarketRegime.BULL_TRENDING
BEAR = MarketRegime.BEAR_TRENDING
T0 = datetime(2026, 1, 5)
NOW = datetime(2026, 1, 10, 12, tzinfo=UTC)


class FakeMarket:
    """Analyze function returning scripted regimes; ``bar`` advances the bar clock."""

    def __init__(self) -> None:
        self.regimes: dict[tuple[str, str], tuple[MarketRegime, float]] = {}
        self.failing: set[tuple[str, str] | str] = set()
        self.bar = 0
        self.calls: list[tuple[str, str]] = []

    def set(self, symbol: str, regime: MarketRegime, confidence: float = 0.9, tfs=None) -> None:
        for tf in tfs or ("1D", "1H", "15m"):
            self.regimes[(symbol, tf)] = (regime, confidence)

    def __call__(self, symbol: str, timeframe: str) -> tuple[RegimeAnalysis, RegimeRecord]:
        self.calls.append((symbol, timeframe))
        if symbol in self.failing or (symbol, timeframe) in self.failing:
            raise ConnectionError(f"provider down for {symbol}")
        regime, confidence = self.regimes[(symbol, timeframe)]
        analysis = RegimeAnalysis(
            current_regime=regime,
            hmm_state=0,
            transition_probability=0.5,
            regime_persistence=0.5,
            recommended_strategy=REGIME_STRATEGIES[regime],
            position_sizing_multiplier=1.0,
            risk_level="Low",
            arbitrage_opportunities=[],
            statistical_signals=[],
            key_levels={},
            regime_confidence=confidence,
        )
        record = RegimeRecord(
            symbol=symbol,
            timeframe=timeframe,
            bar_time=T0 + timedelta(days=self.bar),
            regime=regime.value,
            confidence=confidence,
            persistence=0.5,
            transition_probability=0.5,
            recommended_strategy=REGIME_STRATEGIES[regime].value,
            provider="fake",
            close=100.0 + self.bar,
            recorded_at=NOW,
        )
        return analysis, record


class FakeClock:
    """Monotonic clock advanced only by ``sleep`` (and explicitly by tests)."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


class Collect:
    name = "collect"

    def __init__(self) -> None:
        self.events: list[RegimeChangeEvent] = []

    def send(self, event: RegimeChangeEvent) -> None:
        self.events.append(event)


class Broken:
    name = "broken"

    def send(self, event: RegimeChangeEvent) -> None:
        raise NotifierError("broken delivery failed: HTTP 500")


class Crashing:
    name = "crashing"

    def send(self, event: RegimeChangeEvent) -> None:
        raise RuntimeError("bug in a custom notifier")


@pytest.fixture
def store(tmp_path):
    return SQLiteRegimeStore(tmp_path / "regimes.db")


@pytest.fixture
def market() -> FakeMarket:
    m = FakeMarket()
    m.set("SPY", BEAR)
    m.set("QQQ", BEAR)
    return m


def make_scanner(store, market, notifiers=(), **kwargs) -> Scanner:
    return Scanner(
        kwargs.pop("symbols", ["SPY", "QQQ"]),
        store=store,
        analyze=market,
        notifiers=notifiers,
        clock=lambda: NOW,
        **kwargs,
    )


class TestScanOnce:
    def test_records_saved_and_baseline_not_alerted(self, store, market):
        collect = Collect()
        report = make_scanner(store, market, [collect]).scan_once()
        assert report.symbols_ok == 2 and report.symbols_failed == 0
        assert report.records_saved == 6
        assert report.changes == 0 and not report.events and not collect.events
        assert store.latest("SPY", "1D").regime == BEAR.value
        decisions = report.results[0].decisions
        assert {d.kind for d in decisions.values()} == {ChangeKind.BASELINE}

    def test_confirmed_change_emits_event(self, store, market):
        collect = Collect()
        scanner = make_scanner(store, market, [collect])
        scanner.scan_once()
        market.bar = 1
        market.set("SPY", BULL)
        report = scanner.scan_once(2)

        assert report.changes == 3  # 1D, 1H, 15m all changed; only 1D is watched
        assert report.alerts_sent == 1 and report.alerts_failed == 0
        (event,) = collect.events
        assert (event.symbol, event.timeframe) == ("SPY", "1D")
        assert (event.previous_regime, event.new_regime) == (BEAR.value, BULL.value)
        assert event.confirmation is not None and event.confirmation.confirmed
        assert event.detected_at == NOW
        assert event.recommended_strategy == "Trend Following"
        spy = report.results[0]
        assert spy.decisions["1H"].suppressed is SuppressReason.UNWATCHED
        assert report.results[1].decisions["1D"].kind is ChangeKind.UNCHANGED
        assert "ALERT" in format_scan_report(report)

    def test_rescanning_same_bar_does_not_alert(self, store, market):
        collect = Collect()
        scanner = make_scanner(store, market, [collect])
        scanner.scan_once()
        market.set("SPY", BULL)  # same bar, different regime (e.g. a refit)
        report = scanner.scan_once()
        assert report.changes == 0 and not collect.events
        assert report.results[0].decisions["1D"].kind is ChangeKind.SAME_BAR

    def test_regime_flip_within_a_rescanned_bar_is_not_lost(self, store, market):
        """t0 Bear | t1 Bear | t1 re-scanned Bull -> alert once | t1 Bull again | t2 Bull."""
        collect = Collect()
        scanner = make_scanner(store, market, [collect], symbols=["SPY"])
        scanner.scan_once()
        market.bar = 1
        assert scanner.scan_once().results[0].decisions["1D"].kind is ChangeKind.UNCHANGED

        market.set("SPY", BULL)  # the still-forming bar t1 now classifies as Bull
        report = scanner.scan_once()
        decision = report.results[0].decisions["1D"]
        assert decision.alert and (decision.previous_regime, decision.new_regime) == (
            BEAR.value,
            BULL.value,
        )
        (event,) = collect.events
        assert event.previous_bar_time == T0 and event.bar_time == T0 + timedelta(days=1)

        # Re-scanning the same bar again, and the next bar, never repeat the alert
        assert scanner.scan_once().results[0].decisions["1D"].kind is ChangeKind.SAME_BAR
        market.bar = 2
        assert scanner.scan_once().results[0].decisions["1D"].kind is ChangeKind.UNCHANGED
        assert len(collect.events) == 1

    def test_stale_bar_is_not_saved(self, store, market):
        scanner = make_scanner(store, market, symbols=["SPY"])
        market.bar = 3
        scanner.scan_once()
        market.bar = 1
        market.set("SPY", BULL)
        report = scanner.scan_once()
        spy = report.results[0]
        assert spy.decisions["1D"].kind is ChangeKind.STALE_BAR
        assert spy.saved == ()
        assert [r.bar_time for r in store.history("SPY", timeframe="1D")] == [
            T0 + timedelta(days=3)
        ]
        assert "(stale bar)" in format_scan_report(report)

    def test_unconfirmed_change_is_suppressed(self, store, market):
        collect = Collect()
        scanner = make_scanner(store, market, [collect])
        scanner.scan_once()
        market.bar = 1
        market.set("SPY", BULL, tfs=("1D",))  # lower timeframes stay bearish
        report = scanner.scan_once()
        assert report.changes == 1 and not collect.events
        assert report.results[0].decisions["1D"].suppressed is SuppressReason.NOT_CONFIRMED

    def test_one_symbol_failing_does_not_stop_others(self, store, market):
        market.failing.add("SPY")
        report = make_scanner(store, market).scan_once()
        spy, qqq = report.results
        assert not spy.ok and set(spy.errors) == {"1D", "1H", "15m"}
        assert "provider down" in spy.errors["1D"]
        assert qqq.ok and len(qqq.saved) == 3
        assert (report.symbols_ok, report.symbols_failed) == (1, 1)
        assert not report.all_failed
        assert "1D error: provider down" in format_scan_report(report)

    def test_one_timeframe_failing_keeps_the_others(self, store, market):
        market.failing.add(("SPY", "15m"))
        report = make_scanner(store, market).scan_once()
        spy = report.results[0]
        assert spy.ok and spy.saved == ("1D", "1H") and "15m" in spy.errors
        assert spy.confirmation is not None

    def test_notifier_exceptions_are_isolated_and_counted(self, store, market, caplog):
        collect = Collect()
        scanner = make_scanner(store, market, [Broken(), Crashing(), collect])
        scanner.scan_once()
        market.bar = 1
        market.set("SPY", BULL)
        with caplog.at_level(logging.WARNING, logger="mra_lib.scanner"):
            report = scanner.scan_once()
        assert report.alerts_sent == 1 and report.alerts_failed == 2
        assert len(collect.events) == 1  # the healthy channel still got it
        assert "bug in a custom notifier" in caplog.text

    def test_notifier_error_text_is_secret_masked(self, store, market, caplog):
        secret_url = "https://hooks.example.com/services/T0/B0/s3cretWebhookPath"
        WebhookNotifier(secret_url)  # registers the URL as a secret

        class Leaky:
            name = "leaky"

            def send(self, event):
                raise RuntimeError(f"could not POST to {secret_url}")

        scanner = make_scanner(store, market, [Leaky()])
        scanner.scan_once()
        market.bar = 1
        market.set("SPY", BULL)
        with caplog.at_level(logging.DEBUG):
            report = scanner.scan_once()
        assert report.alerts_failed == 1
        assert "could not POST" in caplog.text
        assert "s3cretWebhookPath" not in caplog.text

    def test_save_failure_skips_detection_for_that_timeframe(self, market):
        class FailingSave(SQLiteRegimeStore):
            def save(self, record):
                if record.timeframe == "1D":
                    raise StorageError("disk full")
                super().save(record)

        store = FailingSave(":memory:")
        report = make_scanner(store, market).scan_once()
        spy = report.results[0]
        assert "1D" not in spy.decisions and "save failed" in spy.errors["1D"]
        assert spy.ok  # analysis succeeded
        assert "(not saved)" in format_scan_report(report)

    def test_store_read_failure_skips_timeframe(self, market):
        class FailingRead(SQLiteRegimeStore):
            def latest(self, symbol, timeframe):
                raise StorageError("locked")

        report = make_scanner(FailingRead(":memory:"), market).scan_once()
        assert report.all_failed
        assert not market.calls  # nothing analyzed without the previous record

    def test_cooldown_survives_restart(self, store, market):
        """A A B (alert) | restart | A -> suppressed by cooldown derived from the store."""
        policy = AlertPolicy(require_confirmation=False)
        collect = Collect()
        for bar, regime in enumerate([BEAR, BEAR, BULL]):
            market.bar = bar
            market.set("SPY", regime)
            make_scanner(store, market, [collect], policy=policy, symbols=["SPY"]).scan_once()
        assert len(collect.events) == 1

        # New process: no in-memory state, the flip back is still suppressed
        market.bar, market.regimes = 3, {}
        market.set("SPY", BEAR)
        report = make_scanner(store, market, [collect], policy=policy, symbols=["SPY"]).scan_once()
        assert report.results[0].decisions["1D"].suppressed is SuppressReason.COOLDOWN
        assert len(collect.events) == 1

        # ...and re-running the same bar after a restart never re-alerts
        report = make_scanner(store, market, [collect], policy=policy, symbols=["SPY"]).scan_once()
        assert report.results[0].decisions["1D"].kind is ChangeKind.SAME_BAR

    def test_stop_skips_remaining_symbols(self, store, market):
        scanner = make_scanner(store, market)
        original = scanner.scan_symbol

        def scan_and_stop(symbol):
            scanner.stop()
            return original(symbol)

        scanner.scan_symbol = scan_and_stop  # type: ignore[method-assign]
        report = scanner.scan_once()
        assert [r.symbol for r in report.results] == ["SPY"]
        assert report.skipped == ("QQQ",)


class TestValidation:
    def test_empty_watchlist(self, store, market):
        with pytest.raises(InvalidParametersError):
            Scanner([], store=store, analyze=market)

    def test_unknown_timeframe(self, store, market):
        with pytest.raises(InvalidParametersError):
            Scanner(["SPY"], store=store, analyze=market, timeframes=["4H"])

    def test_watched_timeframe_must_be_scanned(self, store, market):
        with pytest.raises(InvalidParametersError, match="not scanned"):
            Scanner(["SPY"], store=store, analyze=market, timeframes=["1H", "15m"])

    def test_symbols_normalized_and_deduplicated(self, store, market):
        scanner = Scanner([" spy", "SPY", "qqq"], store=store, analyze=market)
        assert scanner.symbols == ("SPY", "QQQ")
        assert scanner.timeframes == ("1D", "1H", "15m")


class TestRun:
    def test_max_iterations_and_cadence(self, store, market):
        clock = FakeClock()
        original = market.__call__

        def slow_analyze(symbol, tf):
            clock.now += 1.0  # each analysis takes a second: 6 per scan
            return original(symbol, tf)

        scanner = make_scanner(store, slow_analyze)
        summary = scanner.run(60, max_iterations=3, sleep=clock.sleep, monotonic=clock.monotonic)
        assert summary.iterations == 3 and summary.successful_iterations == 3
        assert summary.stop_reason is StopReason.MAX_ITERATIONS
        # Fixed 60s grid: the scan time is subtracted, so the schedule does not drift
        assert clock.sleeps == [54, 54]
        assert summary.records_saved == 18
        assert "3 scan(s) (3 ok, 0 failed" in format_scan_summary(summary)

    def test_backoff_on_failure(self, store, market):
        market.failing.update({"SPY", "QQQ"})
        sleeps: list[float] = []
        summary = make_scanner(store, market).run(
            10, max_iterations=4, max_backoff=25, sleep=sleeps.append
        )
        assert summary.failed_iterations == 4 and summary.successful_iterations == 0
        assert summary.symbol_failures == 8
        assert sleeps == [10, 20, 25]

    def test_recovery_resets_backoff(self, store, market):
        clock = FakeClock()
        market.failing.update({"SPY", "QQQ"})

        def sleep(delay):
            clock.sleep(delay)
            if len(clock.sleeps) == 2:
                market.failing.clear()

        summary = make_scanner(store, market).run(
            10, max_iterations=4, sleep=sleep, monotonic=clock.monotonic
        )
        assert summary.failed_iterations == 2 and summary.successful_iterations == 2
        assert clock.sleeps == [10, 20, 10]  # back on the cadence after a success

    def test_on_report_called_and_errors_contained(self, store, market):
        reports = []

        def on_report(report):
            reports.append(report)
            raise RuntimeError("printing failed")

        summary = make_scanner(store, market).run(
            1, max_iterations=2, on_report=on_report, sleep=lambda d: None
        )
        assert len(reports) == 2 and summary.successful_iterations == 2

    @pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
    def test_stops_on_signal(self, store, market, sig):
        before = signal.getsignal(sig)
        scanner = make_scanner(store, market)
        calls = {"n": 0}
        original = scanner.analyze

        def analyze(symbol, tf):
            calls["n"] += 1
            if calls["n"] == 1:
                signal.raise_signal(sig)
            return original(symbol, tf)

        scanner.analyze = analyze
        summary = scanner.run(1, sleep=lambda d: pytest.fail("should not sleep"))
        assert summary.iterations == 1
        assert summary.stop_reason is StopReason.STOP_REQUESTED
        # The current symbol finished, the rest of the watchlist was skipped
        assert summary.records_saved == 3
        assert signal.getsignal(sig) == before

    def test_second_sigint_aborts(self, store, market):
        scanner = make_scanner(store, market)
        original = scanner.analyze

        def analyze(symbol, tf):
            signal.raise_signal(signal.SIGINT)
            return original(symbol, tf)

        scanner.analyze = analyze
        summary = scanner.run(1)
        assert summary.stop_reason is StopReason.INTERRUPTED


class TestWithMockProvider:
    def test_scan_once_with_mock_provider(self, store, caplog):
        scanner = Scanner(
            ["SPY"],
            store=store,
            provider="mock",
            timeframes=["1D", "1H"],
            notifiers=[LogNotifier()],
        )
        report = scanner.scan_once()
        assert report.symbols_ok == 1 and report.records_saved == 2
        rec = store.latest("SPY", "1D")
        assert rec is not None and rec.provider == "mock"
        # Re-scanning the same bars is never a change
        again = scanner.scan_once()
        assert {d.kind for d in again.results[0].decisions.values()} == {ChangeKind.SAME_BAR}
