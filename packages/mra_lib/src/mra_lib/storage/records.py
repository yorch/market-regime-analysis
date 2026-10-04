"""
Regime history record and the store protocol.

A :class:`RegimeRecord` is one persisted regime classification: the regime the
detector assigned to the last bar of a (symbol, timeframe) series at the time
of the analysis. Records are keyed by ``(symbol, timeframe, bar_time)``, so
re-analyzing the same bar replaces the earlier record instead of adding one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import pandas as pd

from mra_lib.config.symbols import SYMBOL_PATTERN
from mra_lib.errors import StorageInputError

if TYPE_CHECKING:
    from mra_lib.analyzer import MarketRegimeAnalyzer
    from mra_lib.config.data_classes import RegimeAnalysis

# Default and hard maximum number of records returned by ``RegimeStore.history``
DEFAULT_HISTORY_LIMIT = 100
MAX_HISTORY_LIMIT = 1000


def to_naive_utc(value: datetime) -> datetime:
    """Return ``value`` as a tz-naive UTC datetime.

    Aware datetimes are converted to UTC and the tzinfo dropped; naive datetimes
    are assumed to be UTC already (the provider contract for bar timestamps).

    Args:
        value: The datetime to normalize.

    Returns:
        A tz-naive datetime in UTC.

    Raises:
        StorageInputError: If ``value`` is not a datetime.
    """
    if value is pd.NaT:
        raise StorageInputError("Expected a datetime, got NaT")
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise StorageInputError(f"Expected a datetime, got {type(value).__name__}")
    if value.tzinfo is not None:
        try:
            value = value.astimezone(UTC).replace(tzinfo=None)
        except OverflowError as e:
            raise StorageInputError("Datetime is out of range after conversion to UTC") from e
    return value


def normalize_symbol(symbol: str) -> str:
    """Strip, upper-case and validate a ticker symbol.

    Uses the same format as the web API
    (:data:`mra_lib.config.symbols.SYMBOL_PATTERN`), so every stored symbol
    can also be queried over HTTP.

    Raises:
        StorageInputError: If the symbol is empty or has an invalid format.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise StorageInputError("Symbol must be a non-empty string")
    normalized = symbol.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(normalized):
        raise StorageInputError(
            "Invalid symbol: use 1-15 characters (letters, digits, '.', '-', '^', '=')"
        )
    return normalized


def _finite(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise StorageInputError(f"{name} must be a number") from e
    if not math.isfinite(number):
        raise StorageInputError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class RegimeRecord:
    """One persisted regime classification.

    Attributes:
        symbol: Ticker symbol, stored upper-case.
        timeframe: Analysis timeframe (e.g. ``"1D"``, ``"1H"``, ``"15m"``).
        bar_time: Timestamp of the last bar analyzed, tz-naive (the provider
            contract): UTC for intraday bars, the session date at 00:00 for daily
            bars. Aware values are converted to naive UTC on construction.
        regime: The :class:`~mra_lib.config.enums.MarketRegime` value string.
        confidence: Regime confidence (0-1).
        persistence: Regime persistence (0-1).
        transition_probability: Probability of the transition into the current state.
        recommended_strategy: The :class:`~mra_lib.config.enums.TradingStrategy`
            value string.
        provider: Name of the data provider the bars came from.
        close: Close of the last bar analyzed, if known.
        recorded_at: When the record was produced, tz-aware UTC (naive values are
            assumed UTC). Defaults to now.
    """

    symbol: str
    timeframe: str
    bar_time: datetime
    regime: str
    confidence: float
    persistence: float
    transition_probability: float
    recommended_strategy: str
    provider: str
    close: float | None = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Normalize and validate fields.

        Raises:
            StorageInputError: If a field is missing or invalid.
        """
        set_ = object.__setattr__  # frozen dataclass
        set_(self, "symbol", normalize_symbol(self.symbol))
        for name in ("timeframe", "regime", "recommended_strategy", "provider"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise StorageInputError(f"{name} must be a non-empty string")
        set_(self, "bar_time", to_naive_utc(self.bar_time))
        recorded = to_naive_utc(self.recorded_at).replace(tzinfo=UTC)
        set_(self, "recorded_at", recorded)
        for name in ("confidence", "persistence", "transition_probability"):
            set_(self, name, _finite(name, getattr(self, name)))
        if self.close is not None:
            try:
                close = float(self.close)
            except (TypeError, ValueError) as e:
                raise StorageInputError("close must be a number or None") from e
            set_(self, "close", close if math.isfinite(close) else None)

    @classmethod
    def from_analysis(
        cls,
        analysis: RegimeAnalysis,
        analyzer: MarketRegimeAnalyzer,
        timeframe: str,
        *,
        recorded_at: datetime | None = None,
    ) -> RegimeRecord:
        """Build a record from an analysis and the analyzer that produced it.

        ``bar_time`` and ``close`` come from the last bar of
        ``analyzer.data[timeframe]``; ``provider`` from the analyzer's provider.

        Args:
            analysis: Result of ``analyzer.analyze_current_regime(timeframe)``.
            analyzer: The analyzer holding the bars the analysis was computed on.
            timeframe: The analyzed timeframe.
            recorded_at: Override the record timestamp (default: now, UTC).

        Returns:
            The record.

        Raises:
            StorageInputError: If the analyzer holds no bars for ``timeframe``.
        """
        df = analyzer.data.get(timeframe)
        if df is None or df.empty:
            raise StorageInputError(f"No bars loaded for timeframe {timeframe}")
        last = pd.Timestamp(df.index[-1])
        if last is pd.NaT:
            raise StorageInputError(f"Last bar of timeframe {timeframe} has no timestamp")
        bar_time = cast(datetime, last.to_pydatetime())
        close: float | None = None
        if "Close" in df.columns:
            close = float(df["Close"].iloc[-1])
        return cls(
            symbol=analyzer.symbol,
            timeframe=timeframe,
            bar_time=bar_time,
            regime=analysis.current_regime.value,
            confidence=analysis.regime_confidence,
            persistence=analysis.regime_persistence,
            transition_probability=analysis.transition_probability,
            recommended_strategy=analysis.recommended_strategy.value,
            provider=analyzer.provider.provider_name,
            close=close,
            recorded_at=recorded_at if recorded_at is not None else datetime.now(UTC),
        )


@runtime_checkable
class RegimeStore(Protocol):
    """Persistence interface for regime history.

    Implementations must be safe to call from multiple threads.
    """

    def save(self, record: RegimeRecord) -> None:
        """Insert ``record``, replacing any record for the same (symbol, timeframe, bar_time)."""
        ...

    def latest(self, symbol: str, timeframe: str) -> RegimeRecord | None:
        """Return the record with the newest ``bar_time`` for (symbol, timeframe), if any."""
        ...

    def history(
        self,
        symbol: str,
        timeframe: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> list[RegimeRecord]:
        """Return records for ``symbol``, newest ``bar_time`` first.

        ``since``/``until`` bound ``bar_time`` inclusively; ``limit`` is capped
        at :data:`MAX_HISTORY_LIMIT`.
        """
        ...

    def symbols(self) -> list[str]:
        """Return every symbol with at least one record, sorted."""
        ...
