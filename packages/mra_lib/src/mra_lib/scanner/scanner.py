"""
Watchlist scanner: analyze, record, detect regime changes, alert.

One :meth:`Scanner.scan_once` processes every symbol in the watchlist. Per
symbol, in this order:

1. read the previous :meth:`~mra_lib.storage.RegimeStore.latest` record for
   every timeframe, **before** anything is saved;
2. analyze each timeframe (each in its own analyzer, so one failing timeframe
   does not hide the others);
3. save a :class:`~mra_lib.storage.RegimeRecord` for every timeframe that
   succeeded (for a re-scan of a stored bar, the newest record *before* that bar
   is read first, as the comparison baseline; a bar older than the newest stored
   one is not saved);
4. compute the multi-timeframe confirmation from the successful analyses;
5. detect changes (:func:`~mra_lib.scanner.detection.detect_change`), reading
   the stored history for the cooldown only when a change is found;
6. send a :class:`~mra_lib.scanner.events.RegimeChangeEvent` per alert to every
   notifier.

Failure isolation: an exception for one symbol (or timeframe) is logged and
counted, and the scan moves on. A timeframe whose record could not be saved is
not checked for changes: the next scan would see the same change again, so
alerting now could alert twice. Notifier exceptions are logged (secrets masked)
and counted, never raised.

Delivery is at most once: records are saved *before* alerts are sent, so a
crash between the two loses those alerts rather than repeating them after a
restart.

:meth:`Scanner.run` repeats scans on a monotonic cadence via
:func:`mra_lib.scheduling.run_periodic` (exponential backoff when a whole scan
fails, clean stop on SIGINT/SIGTERM) and returns a :class:`ScanSummary`.
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.timeframes import DEFAULT_PERIODS, TIMEFRAMES
from mra_lib.errors import InvalidParametersError
from mra_lib.indicators.base import RegimeDetector
from mra_lib.scheduling import DEFAULT_MAX_BACKOFF, StopReason, StopToken, run_periodic
from mra_lib.signals.confirmation import TimeframeConfirmation, confirm_timeframes
from mra_lib.storage import RegimeRecord, RegimeStore
from mra_lib.storage.records import normalize_symbol

from .detection import AlertPolicy, ChangeDecision, ChangeKind, detect_change
from .events import RegimeChangeEvent
from .notifiers import Notifier, describe_notifier, mask_secrets

logger = logging.getLogger(__name__)

#: Analyze one (symbol, timeframe): returns the analysis and the record to store.
AnalyzeFn = Callable[[str, str], tuple[RegimeAnalysis, RegimeRecord]]

_ERROR_TEXT_LIMIT = 200


def _error_text(exc: BaseException) -> str:
    """Short, secret-masked description of an exception."""
    text = str(exc) or type(exc).__name__
    return mask_secrets(text)[:_ERROR_TEXT_LIMIT]


class ProviderAnalyzer:
    """Default :data:`AnalyzeFn`: fetch, fit and analyze one timeframe via a provider.

    Each call builds a :class:`~mra_lib.analyzer.MarketRegimeAnalyzer` for a
    single timeframe. Provider rate limits are shared per provider class (see
    :class:`~mra_lib.data_providers.base.MarketDataProvider`), so many calls in a
    row are throttled correctly.

    Args:
        provider: Registered provider name (e.g. ``"yfinance"``, ``"mock"``).
        api_key: Provider API key, if the provider needs one.
        periods: ``timeframe -> period`` overrides (default ``DEFAULT_PERIODS``).
        detector_factory: Optional detector factory passed to the analyzer.
        clock: Returns the ``recorded_at`` timestamp (default: now, UTC).
    """

    def __init__(
        self,
        provider: str = "yfinance",
        api_key: str | None = None,
        *,
        periods: Mapping[str, str] | None = None,
        detector_factory: Callable[[], RegimeDetector] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.provider = provider
        self._api_key = api_key
        self.periods = {**DEFAULT_PERIODS, **(periods or {})}
        self.detector_factory = detector_factory
        self._clock = clock or (lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(provider={self.provider!r})"

    def __call__(self, symbol: str, timeframe: str) -> tuple[RegimeAnalysis, RegimeRecord]:
        """Analyze ``timeframe`` of ``symbol`` and build its record."""
        from mra_lib.analyzer import MarketRegimeAnalyzer  # noqa: PLC0415 - heavy import

        analyzer = MarketRegimeAnalyzer(
            symbol,
            periods={timeframe: self.periods[timeframe]},
            provider_flag=self.provider,
            api_key=self._api_key,
            detector_factory=self.detector_factory,
        )
        analysis = analyzer.analyze_current_regime(timeframe)
        record = RegimeRecord.from_analysis(
            analysis, analyzer, timeframe, recorded_at=self._clock()
        )
        return analysis, record


@dataclass(frozen=True)
class SymbolScanResult:
    """Outcome of scanning one symbol.

    Attributes:
        symbol: The symbol.
        records: Records produced this scan, by timeframe (saved or not).
        saved: Timeframes whose record was saved.
        errors: ``timeframe -> error text`` for failed timeframes (analysis,
            store read or save); ``"*"`` for a symbol-level failure.
        decisions: Change decisions, by timeframe.
        confirmation: The multi-timeframe confirmation (``None`` if nothing analyzed).
        events: Alert events emitted.
        alerts_sent: Successful (event, notifier) deliveries.
        alerts_failed: Failed (event, notifier) deliveries.
    """

    symbol: str
    records: dict[str, RegimeRecord] = field(default_factory=dict)
    saved: tuple[str, ...] = ()
    errors: dict[str, str] = field(default_factory=dict)
    decisions: dict[str, ChangeDecision] = field(default_factory=dict)
    confirmation: TimeframeConfirmation | None = None
    events: tuple[RegimeChangeEvent, ...] = ()
    alerts_sent: int = 0
    alerts_failed: int = 0

    @property
    def ok(self) -> bool:
        """Whether at least one timeframe was analyzed."""
        return bool(self.records)

    @property
    def changes(self) -> int:
        """Number of regime changes detected (alerted or suppressed)."""
        return sum(1 for d in self.decisions.values() if d.changed)


@dataclass(frozen=True)
class ScanReport:
    """Outcome of one :meth:`Scanner.scan_once`.

    Attributes:
        iteration: 1-based iteration number (1 for a standalone scan).
        started_at: When the scan started (aware UTC).
        duration: Wall-clock seconds the scan took.
        results: Per-symbol results, in watchlist order.
        skipped: Symbols not scanned because a stop was requested.
    """

    iteration: int
    started_at: datetime
    duration: float
    results: tuple[SymbolScanResult, ...]
    skipped: tuple[str, ...] = ()

    @property
    def symbols_ok(self) -> int:
        """Symbols with at least one analyzed timeframe."""
        return sum(1 for r in self.results if r.ok)

    @property
    def symbols_failed(self) -> int:
        """Symbols where every timeframe failed."""
        return sum(1 for r in self.results if not r.ok)

    @property
    def records_saved(self) -> int:
        """Records written to the store."""
        return sum(len(r.saved) for r in self.results)

    @property
    def changes(self) -> int:
        """Regime changes detected (alerted or suppressed)."""
        return sum(r.changes for r in self.results)

    @property
    def events(self) -> tuple[RegimeChangeEvent, ...]:
        """Every alert event emitted, in watchlist order."""
        return tuple(e for r in self.results for e in r.events)

    @property
    def alerts_sent(self) -> int:
        """Successful (event, notifier) deliveries."""
        return sum(r.alerts_sent for r in self.results)

    @property
    def alerts_failed(self) -> int:
        """Failed (event, notifier) deliveries."""
        return sum(r.alerts_failed for r in self.results)

    @property
    def all_failed(self) -> bool:
        """Whether symbols were scanned and every one of them failed."""
        return bool(self.results) and self.symbols_ok == 0


@dataclass
class ScanSummary:
    """Totals over a :meth:`Scanner.run`.

    Attributes:
        iterations: Scans started.
        successful_iterations: Scans where at least one symbol succeeded.
        failed_iterations: Scans where every symbol failed, or that raised.
        symbol_failures: Total failed symbol scans.
        records_saved: Total records written.
        changes: Total regime changes detected.
        alerts: Total alert events emitted.
        alerts_sent: Total successful deliveries.
        alerts_failed: Total failed deliveries.
        stop_reason: Why the run ended (``None`` while running).
    """

    iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    symbol_failures: int = 0
    records_saved: int = 0
    changes: int = 0
    alerts: int = 0
    alerts_sent: int = 0
    alerts_failed: int = 0
    stop_reason: StopReason | None = None

    def add(self, report: ScanReport) -> None:
        """Add one scan's counts (the iteration counters are kept by :meth:`Scanner.run`)."""
        self.symbol_failures += report.symbols_failed
        self.records_saved += report.records_saved
        self.changes += report.changes
        self.alerts += len(report.events)
        self.alerts_sent += report.alerts_sent
        self.alerts_failed += report.alerts_failed


def _describe_decision(decision: ChangeDecision) -> str:
    if decision.kind is ChangeKind.CHANGED:
        status = "ALERT" if decision.alert else f"suppressed: {decision.suppressed}"
        return f"{decision.previous_regime} -> {decision.new_regime} ({status})"
    if decision.kind is ChangeKind.UNCHANGED:
        return decision.new_regime
    return f"{decision.new_regime} ({decision.kind.value.replace('_', ' ')})"


def format_scan_report(report: ScanReport, *, details: bool = True) -> str:
    """Format a scan as a short, human-readable summary.

    Args:
        report: The scan.
        details: Add one line per symbol (regimes, decisions, errors).

    Returns:
        Text without a trailing newline.
    """
    head = (
        f"Scan #{report.iteration} ({report.duration:.1f}s): "
        f"{report.symbols_ok} symbol(s) ok, {report.symbols_failed} failed | "
        f"{report.records_saved} record(s) saved | {report.changes} change(s) | "
        f"{len(report.events)} alert(s): {report.alerts_sent} sent, "
        f"{report.alerts_failed} failed"
    )
    lines = [head]
    if report.skipped:
        lines.append(f"  stopped early; skipped: {', '.join(report.skipped)}")
    if not details:
        return "\n".join(lines)
    for result in report.results:
        parts = [
            f"{tf} {_describe_decision(result.decisions[tf])}"
            for tf in TIMEFRAMES
            if tf in result.decisions
        ]
        parts += [
            f"{tf} {result.records[tf].regime} (not saved)"
            for tf in TIMEFRAMES
            if tf in result.records and tf not in result.decisions
        ]
        conf = result.confirmation
        if conf is not None:
            state = "confirmed" if conf.confirmed else conf.reason.value
            parts.append(f"confirmation {conf.direction.value} {state}")
        parts += [
            f"{tf} error: {msg}" if tf != "*" else f"error: {msg}"
            for tf, msg in result.errors.items()
        ]
        lines.append(f"  {result.symbol}: " + ("; ".join(parts) or "nothing analyzed"))
    return "\n".join(lines)


def format_scan_summary(summary: ScanSummary) -> str:
    """Format a :class:`ScanSummary` as one line."""
    reason = f", stopped: {summary.stop_reason.value}" if summary.stop_reason else ""
    return (
        f"Scanner finished: {summary.iterations} scan(s) "
        f"({summary.successful_iterations} ok, {summary.failed_iterations} failed{reason}) | "
        f"{summary.symbol_failures} symbol failure(s) | {summary.records_saved} record(s) saved"
        f" | {summary.changes} change(s) | {summary.alerts} alert(s): "
        f"{summary.alerts_sent} sent, {summary.alerts_failed} failed"
    )


class Scanner:
    """Periodically analyze a watchlist, record regimes and alert on changes.

    Args:
        symbols: Watchlist (normalized to upper case; duplicates dropped).
        store: Where records are read from and saved to.
        timeframes: Timeframes to analyze (default: all of ``TIMEFRAMES``).
        provider: Data provider name for the default analyzer.
        api_key: Provider API key for the default analyzer.
        periods: ``timeframe -> period`` overrides for the default analyzer.
        notifiers: Alert channels (default: none; the events are still reported).
        policy: Alert policy (default :class:`AlertPolicy()`).
        analyze: Custom ``(symbol, timeframe) -> (analysis, record)`` function;
            replaces the provider-based default (useful for tests).
        clock: Returns "now" (aware UTC) for ``detected_at``.

    Raises:
        InvalidParametersError: On an empty watchlist, an unknown timeframe, or a
            watched timeframe that is not scanned.
        StorageInputError: On an invalid symbol.
    """

    def __init__(  # noqa: PLR0913 - keyword-only configuration
        self,
        symbols: Iterable[str],
        *,
        store: RegimeStore,
        timeframes: Sequence[str] | None = None,
        provider: str = "yfinance",
        api_key: str | None = None,
        periods: Mapping[str, str] | None = None,
        notifiers: Iterable[Notifier] = (),
        policy: AlertPolicy | None = None,
        analyze: AnalyzeFn | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.symbols = tuple(dict.fromkeys(normalize_symbol(s) for s in symbols))
        if not self.symbols:
            raise InvalidParametersError("The watchlist must contain at least one symbol")
        requested = tuple(dict.fromkeys(timeframes if timeframes is not None else TIMEFRAMES))
        unknown = [tf for tf in requested if tf not in TIMEFRAMES]
        if not requested or unknown:
            raise InvalidParametersError(
                f"timeframes must be a non-empty subset of {list(TIMEFRAMES)}, "
                f"got {list(requested)}"
            )
        # Coarsest first, like the rest of the library
        self.timeframes = tuple(tf for tf in TIMEFRAMES if tf in requested)
        self.policy = policy if policy is not None else AlertPolicy()
        unscanned = [tf for tf in self.policy.timeframes_to_watch if tf not in self.timeframes]
        if unscanned:
            raise InvalidParametersError(
                f"Watched timeframe(s) {unscanned} are not scanned ({list(self.timeframes)})"
            )
        if self.policy.require_confirmation and len(self.timeframes) < 2:  # noqa: PLR2004
            logger.warning(
                "Confirmation needs at least two timeframes; with only %s no change will alert "
                "(disable require_confirmation or scan more timeframes)",
                ", ".join(self.timeframes),
            )
        self.store = store
        self.notifiers: tuple[Notifier, ...] = tuple(notifiers)
        self._clock = clock or (lambda: datetime.now(UTC))
        self.analyze: AnalyzeFn = analyze or ProviderAnalyzer(
            provider, api_key, periods=periods, clock=self._clock
        )
        self._stop = StopToken()

    def __repr__(self) -> str:
        channels = ", ".join(describe_notifier(n) for n in self.notifiers) or "none"
        return (
            f"{type(self).__name__}(symbols={list(self.symbols)}, "
            f"timeframes={list(self.timeframes)}, notifiers=[{channels}])"
        )

    def stop(self) -> None:
        """Ask a running :meth:`run` / :meth:`scan_once` to stop after the current symbol.

        The request stays set for later standalone :meth:`scan_once` calls (they
        skip every symbol); :meth:`run` starts with a fresh stop flag.
        """
        self._stop.request()

    # ── one symbol ───────────────────────────────────────────────────────

    def _read_previous(self, symbol: str, errors: dict[str, str]) -> dict[str, RegimeRecord | None]:
        previous: dict[str, RegimeRecord | None] = {}
        for tf in self.timeframes:
            try:
                previous[tf] = self.store.latest(symbol, tf)
            except Exception as e:  # noqa: BLE001 - isolate: skip this timeframe
                errors[tf] = f"store read failed: {_error_text(e)}"
                logger.warning(
                    "Scanner: reading %s %s from the store failed: %s", symbol, tf, _error_text(e)
                )
        return previous

    def _references(
        self, latest: RegimeRecord | None, current: RegimeRecord
    ) -> tuple[RegimeRecord | None, RegimeRecord | None]:
        """Return ``(previous, replaced)`` for :func:`detect_change`.

        For a new bar, ``previous`` is the latest stored record. For a re-scan of
        the latest stored bar, ``previous`` is the newest record *before* that bar
        and ``replaced`` the stored record the new one overwrites.
        """
        if latest is None or current.bar_time > latest.bar_time:
            return latest, None
        older = self.store.history(
            current.symbol,
            timeframe=current.timeframe,
            until=latest.bar_time - timedelta(microseconds=1),
            limit=1,
        )
        return (older[0] if older else None), latest

    def _analyze(
        self, symbol: str, timeframes: Iterable[str], errors: dict[str, str]
    ) -> tuple[dict[str, RegimeAnalysis], dict[str, RegimeRecord]]:
        analyses: dict[str, RegimeAnalysis] = {}
        records: dict[str, RegimeRecord] = {}
        for tf in timeframes:
            try:
                analysis, record = self.analyze(symbol, tf)
            except Exception as e:  # noqa: BLE001 - isolate: other timeframes still run
                errors[tf] = _error_text(e)
                logger.warning("Scanner: analysis of %s %s failed: %s", symbol, tf, _error_text(e))
                continue
            analyses[tf] = analysis
            records[tf] = record
        return analyses, records

    def _save(
        self, symbol: str, records: Mapping[str, RegimeRecord], errors: dict[str, str]
    ) -> tuple[str, ...]:
        saved: list[str] = []
        for tf, record in records.items():
            try:
                self.store.save(record)
                saved.append(tf)
            except Exception as e:  # noqa: BLE001 - isolate: no change detection for tf
                errors[tf] = f"save failed: {_error_text(e)}"
                logger.warning("Scanner: saving %s %s failed: %s", symbol, tf, _error_text(e))
        return tuple(saved)

    def _cooldown_history(
        self, previous: RegimeRecord, current: RegimeRecord
    ) -> Sequence[RegimeRecord]:
        """Stored history for the cooldown check, read only for a watched change."""
        if (
            not self.policy.uses_cooldown
            or current.bar_time <= previous.bar_time
            or current.regime == previous.regime
            or current.timeframe not in self.policy.timeframes_to_watch
        ):
            return ()
        try:
            return self.store.history(
                previous.symbol,
                timeframe=previous.timeframe,
                until=previous.bar_time,
                limit=self.policy.history_lookback,
            )
        except Exception as e:  # noqa: BLE001 - fall back to "no visible transition"
            logger.warning(
                "Scanner: reading history for the %s %s cooldown failed (%s); "
                "evaluating without cooldown",
                previous.symbol,
                previous.timeframe,
                _error_text(e),
            )
            return ()

    def _notify(self, event: RegimeChangeEvent) -> tuple[int, int]:
        sent = failed = 0
        for notifier in self.notifiers:
            try:
                notifier.send(event)
                sent += 1
            except Exception as e:  # noqa: BLE001 - a broken channel never stops the scan
                failed += 1
                logger.warning(
                    "Scanner: %s alert for %s %s failed: %s",
                    getattr(notifier, "name", type(notifier).__name__),
                    event.symbol,
                    event.timeframe,
                    _error_text(e),
                )
        return sent, failed

    def scan_symbol(self, symbol: str) -> SymbolScanResult:
        """Scan one symbol (steps 1-6 in the module docstring); never raises.

        Args:
            symbol: Ticker symbol.

        Returns:
            The symbol's result.
        """
        try:
            symbol = normalize_symbol(symbol)
        except Exception as e:  # noqa: BLE001 - reported as a symbol-level error
            return SymbolScanResult(symbol=str(symbol), errors={"*": _error_text(e)})
        errors: dict[str, str] = {}
        try:
            # 1. Previous records, before anything is saved
            previous = self._read_previous(symbol, errors)
            # 2. Analyze (timeframes whose previous record is unknown are skipped)
            analyses, records = self._analyze(symbol, previous, errors)
            # 3. Save (a bar older than the newest stored one is not saved: it
            #    would rewrite the history the cooldown reads)
            decisions: dict[str, ChangeDecision] = {}
            refs: dict[str, tuple[RegimeRecord | None, RegimeRecord | None]] = {}
            to_save: dict[str, RegimeRecord] = {}
            for tf, current in records.items():
                latest = previous[tf]
                if latest is not None and current.bar_time < latest.bar_time:
                    decisions[tf] = detect_change(latest, current, policy=self.policy)
                    continue
                try:
                    refs[tf] = self._references(latest, current)
                except Exception as e:  # noqa: BLE001 - isolate: skip this timeframe
                    errors[tf] = f"store read failed: {_error_text(e)}"
                    logger.warning(
                        "Scanner: reading %s %s history failed: %s", symbol, tf, _error_text(e)
                    )
                    continue
                to_save[tf] = current
            saved = self._save(symbol, to_save, errors)
            # 4. Confirmation
            confirmation = confirm_timeframes(analyses) if analyses else None
            # 5. Detect
            events: list[RegimeChangeEvent] = []
            for tf in saved:
                (prev, replaced), current = refs[tf], records[tf]
                history = self._cooldown_history(prev, current) if prev is not None else ()
                decision = detect_change(
                    prev,
                    current,
                    policy=self.policy,
                    confirmation=confirmation,
                    history=history,
                    replaced=replaced,
                )
                decisions[tf] = decision
                if decision.changed:
                    logger.info(
                        "Scanner: %s %s regime change %s -> %s (%s)",
                        symbol,
                        tf,
                        decision.previous_regime,
                        decision.new_regime,
                        "alert" if decision.alert else f"suppressed: {decision.suppressed}",
                    )
                if decision.alert and prev is not None:
                    events.append(
                        RegimeChangeEvent.from_records(
                            prev, current, confirmation, detected_at=self._clock()
                        )
                    )
            # 6. Notify
            sent = failed = 0
            for event in events:
                s, f = self._notify(event)
                sent += s
                failed += f
        except Exception as e:  # noqa: BLE001 - one symbol never stops the scan
            errors["*"] = _error_text(e)
            # No traceback: its frames' text could carry secrets
            logger.error(
                "Scanner: unexpected failure scanning %s: %s: %s",
                symbol,
                type(e).__name__,
                errors["*"],
            )
            return SymbolScanResult(symbol=symbol, errors=errors)

        return SymbolScanResult(
            symbol=symbol,
            records=records,
            saved=saved,
            errors=errors,
            decisions=decisions,
            confirmation=confirmation,
            events=tuple(events),
            alerts_sent=sent,
            alerts_failed=failed,
        )

    # ── whole watchlist ──────────────────────────────────────────────────

    def scan_once(self, iteration: int = 1) -> ScanReport:
        """Scan every symbol once; never raises for per-symbol failures.

        Stops early (remaining symbols listed in ``ScanReport.skipped``) if
        :meth:`stop` is called or a stop signal arrives during :meth:`run`.

        Args:
            iteration: Iteration number reported in the result.

        Returns:
            The scan report.
        """
        started_at = self._clock()
        start = time.monotonic()
        results: list[SymbolScanResult] = []
        skipped: tuple[str, ...] = ()
        for index, symbol in enumerate(self.symbols):
            if self._stop.requested:
                skipped = self.symbols[index:]
                break
            result = self.scan_symbol(symbol)
            if not result.ok:
                logger.warning(
                    "Scanner: %s failed: %s",
                    symbol,
                    "; ".join(f"{tf}: {msg}" for tf, msg in result.errors.items()),
                )
            results.append(result)
        return ScanReport(
            iteration=iteration,
            started_at=started_at,
            duration=time.monotonic() - start,
            results=tuple(results),
            skipped=skipped,
        )

    def run(  # noqa: PLR0913 - keyword-only knobs with defaults
        self,
        interval: float,
        max_iterations: int | None = None,
        *,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        on_report: Callable[[ScanReport], None] | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        signals: Iterable[signal.Signals] = (signal.SIGINT, signal.SIGTERM),
    ) -> ScanSummary:
        """Scan every ``interval`` seconds until stopped.

        A scan where every symbol failed (or that raised) counts as a failed
        iteration and is retried with exponential backoff (``interval``,
        ``2*interval``, ... capped at ``max(max_backoff, interval)``). The first
        SIGINT/SIGTERM stops after the current symbol; a second SIGINT aborts
        immediately (``KeyboardInterrupt``, still caught and summarized).

        Args:
            interval: Seconds between scans (monotonic cadence).
            max_iterations: Stop after this many scans (``None`` = until stopped).
            max_backoff: Upper bound in seconds for the retry delay.
            on_report: Called with each :class:`ScanReport` (exceptions are logged);
                without it, a one-line summary of each scan is logged at INFO.
            sleep: Sleep function between scans (injectable for tests).
            monotonic: Clock for the cadence (injectable for tests).
            signals: Signals that request a stop (handlers only installed from the
                main thread, restored on exit).

        Returns:
            Totals over all scans, including why the run ended.
        """
        summary = ScanSummary()
        self._stop = StopToken()

        def step(iteration: int) -> bool:
            report = self.scan_once(iteration)
            summary.add(report)
            if on_report is not None:
                try:
                    on_report(report)
                except Exception as e:  # noqa: BLE001 - reporting must not break the loop
                    logger.error("Scanner: on_report callback failed: %s", _error_text(e))
            if on_report is None:
                logger.info("%s", format_scan_report(report, details=False))
            if report.all_failed:
                raise RuntimeError(f"every symbol failed ({report.symbols_failed})")
            return True

        logger.info(
            "Scanner started: %s on %s (%s), alerts via %s",
            ", ".join(self.symbols),
            ", ".join(self.timeframes),
            "single scan" if max_iterations == 1 else f"every {interval:g}s",
            ", ".join(describe_notifier(n) for n in self.notifiers) or "no channel",
        )
        result = run_periodic(
            step,
            interval,
            max_iterations=max_iterations,
            max_backoff=max_backoff,
            stop=self._stop,
            sleep=sleep,
            monotonic=monotonic,
            signals=signals,
            name="Scan",
            log=logger,
        )
        summary.iterations = result.iterations
        summary.successful_iterations = result.successes
        summary.failed_iterations = result.failures
        summary.stop_reason = result.stop_reason
        logger.info("%s", format_scan_summary(summary))
        return summary


__all__ = [
    "AnalyzeFn",
    "ProviderAnalyzer",
    "ScanReport",
    "ScanSummary",
    "Scanner",
    "SymbolScanResult",
    "format_scan_report",
    "format_scan_summary",
]
