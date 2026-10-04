"""
The regime-change alert event sent to notifiers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from mra_lib.signals.confirmation import TimeframeConfirmation
from mra_lib.storage import RegimeRecord

#: ``event`` field of :meth:`RegimeChangeEvent.to_dict`.
EVENT_TYPE = "regime_change"


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class RegimeChangeEvent:
    """A regime change that passed the alert policy.

    Attributes:
        symbol: Ticker symbol (upper-case).
        timeframe: Timeframe whose regime changed.
        previous_regime: The previously stored regime.
        new_regime: The new regime.
        confidence: Confidence (0-1) of the new regime.
        bar_time: Bar the new regime was detected on (naive UTC, as stored).
        previous_bar_time: Bar of the previously stored record.
        recommended_strategy: Strategy recommended for the new regime.
        confirmation: Multi-timeframe confirmation from the same scan, if any.
        detected_at: When the scanner detected the change (aware UTC).
        provider: Data provider the bars came from.
        close: Close of the bar, if known.
    """

    symbol: str
    timeframe: str
    previous_regime: str
    new_regime: str
    confidence: float
    bar_time: datetime
    previous_bar_time: datetime
    recommended_strategy: str
    confirmation: TimeframeConfirmation | None = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    provider: str = ""
    close: float | None = None

    @classmethod
    def from_records(
        cls,
        previous: RegimeRecord,
        current: RegimeRecord,
        confirmation: TimeframeConfirmation | None = None,
        *,
        detected_at: datetime | None = None,
    ) -> RegimeChangeEvent:
        """Build the event for a change from ``previous`` to ``current``.

        Args:
            previous: The stored record before the change.
            current: The new record.
            confirmation: The scan's multi-timeframe confirmation.
            detected_at: Detection time (default: now, UTC).

        Returns:
            The event.
        """
        return cls(
            symbol=current.symbol,
            timeframe=current.timeframe,
            previous_regime=previous.regime,
            new_regime=current.regime,
            confidence=current.confidence,
            bar_time=current.bar_time,
            previous_bar_time=previous.bar_time,
            recommended_strategy=current.recommended_strategy,
            confirmation=confirmation,
            detected_at=detected_at if detected_at is not None else datetime.now(UTC),
            provider=current.provider,
            close=current.close,
        )

    @property
    def title(self) -> str:
        """One-line headline, e.g. ``SPY 1D: Bear Trending -> Bull Trending``."""
        return f"{self.symbol} {self.timeframe}: {self.previous_regime} -> {self.new_regime}"

    def confirmation_summary(self) -> str:
        """Short confirmation text, e.g. ``bullish, confirmed (agreement 68%; 1D, 1H)``."""
        conf = self.confirmation
        if conf is None:
            return "not computed"
        status = "confirmed" if conf.confirmed else f"not confirmed ({conf.reason.value})"
        aligned = ", ".join(conf.aligned_timeframes) or "none"
        return f"{conf.direction.value}, {status} (agreement {conf.agreement:.0%}; {aligned})"

    def format_message(self) -> str:
        """Short plain-text alert message (a few lines, no markup)."""
        lines = [
            f"Regime change: {self.title}",
            f"Confidence {self.confidence:.0%} | bar {self.bar_time.isoformat(sep=' ')} UTC"
            + (f" | close {self.close:.2f}" if self.close is not None else ""),
            f"Confirmation: {self.confirmation_summary()}",
            f"Strategy: {self.recommended_strategy}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (ISO 8601 timestamps, non-finite floats as ``None``)."""
        return {
            "event": EVENT_TYPE,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "previous_regime": self.previous_regime,
            "new_regime": self.new_regime,
            "confidence": _finite_or_none(self.confidence),
            "bar_time": self.bar_time.isoformat(),
            "previous_bar_time": self.previous_bar_time.isoformat(),
            "recommended_strategy": self.recommended_strategy,
            "confirmation": self.confirmation.to_dict() if self.confirmation else None,
            "detected_at": self.detected_at.isoformat(),
            "provider": self.provider,
            "close": _finite_or_none(self.close),
            "message": self.format_message(),
        }


__all__ = ["EVENT_TYPE", "RegimeChangeEvent"]
