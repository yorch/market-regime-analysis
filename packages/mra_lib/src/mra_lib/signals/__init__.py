"""Signals derived from regime analyses (e.g. multi-timeframe confirmation)."""

from .confirmation import (
    ConfirmationReason,
    TimeframeConfirmation,
    confirm_analyzer_timeframes,
    confirm_timeframes,
    format_confirmation_report,
)

__all__ = [
    "ConfirmationReason",
    "TimeframeConfirmation",
    "confirm_analyzer_timeframes",
    "confirm_timeframes",
    "format_confirmation_report",
]
