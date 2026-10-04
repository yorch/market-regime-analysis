"""
Multi-timeframe confirmation signal.

Combines per-timeframe :class:`~mra_lib.config.data_classes.RegimeAnalysis`
results into one :class:`TimeframeConfirmation` that says whether the
timeframes agree on a direction. The core function, :func:`confirm_timeframes`,
is pure: no I/O, no logging side effects, no analyzer construction, and the same
input always yields the same output.

Signal design
-------------
1. **Bias.** Each timeframe's regime is mapped to a :class:`DirectionalBias`
   via :data:`~mra_lib.config.regime_tables.REGIME_BIAS`. ``UNKNOWN`` regimes
   are treated exactly like missing timeframes ("unavailable").
2. **Primary timeframe.** The highest (coarsest) available timeframe, in
   :data:`~mra_lib.config.timeframes.TIMEFRAMES` order (1D > 1H > 15m). Its
   bias is the signal's ``direction``: higher timeframes dominate.
3. **Agreement.** The confidence-weighted share of the *total configured
   weight* whose bias matches ``direction``::

       agreement = sum(weight[tf] * confidence[tf] for tf in aligned) / sum(weight.values())

   Unavailable timeframes keep their weight in the denominator, so missing or
   UNKNOWN timeframes lower agreement. Conflicting and neutral timeframes add
   nothing to the numerator.
4. **Confirmation.** ``confirmed`` is true only if at least
   :data:`~mra_lib.config.timeframes.MIN_CONFIRMATION_TIMEFRAMES` timeframes are
   available, the primary timeframe is directional (bullish/bearish), at least
   that many timeframes are aligned with it (the primary plus a lower timeframe),
   and ``agreement >= threshold``.

``HIGH_VOLATILITY`` is neutral in direction and reported separately in
``risk_timeframes`` (see :data:`~mra_lib.config.regime_tables.RISK_REGIMES`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.enums import DirectionalBias, MarketRegime
from mra_lib.config.regime_tables import REGIME_BIAS, RISK_REGIMES
from mra_lib.config.timeframes import (
    CONFIRMATION_THRESHOLD,
    CONFIRMATION_WEIGHTS,
    MIN_CONFIRMATION_TIMEFRAMES,
    TIMEFRAMES,
)

if TYPE_CHECKING:
    from mra_lib.analyzer import MarketRegimeAnalyzer

_PRECISION = 9
"""Decimal places kept for ``agreement``/``confidence`` (stable, float-noise-free output)."""

_OPPOSITE = {
    DirectionalBias.BULLISH: DirectionalBias.BEARISH,
    DirectionalBias.BEARISH: DirectionalBias.BULLISH,
}


class ConfirmationReason(StrEnum):
    """Machine-readable explanation of a :class:`TimeframeConfirmation` outcome.

    Checked in this order; the first that applies is reported.
    """

    NO_TIMEFRAMES = "no_timeframes"
    """No timeframe has a classified (non-UNKNOWN) regime."""

    INSUFFICIENT_TIMEFRAMES = "insufficient_timeframes"
    """Fewer than ``MIN_CONFIRMATION_TIMEFRAMES`` timeframes are available."""

    PRIMARY_NEUTRAL = "primary_neutral"
    """The primary timeframe has no directional bias."""

    NO_ALIGNMENT = "no_alignment"
    """No lower timeframe agrees with the primary timeframe's direction."""

    BELOW_THRESHOLD = "below_threshold"
    """Timeframes agree, but the agreement score is below the threshold."""

    CONFIRMED = "confirmed"
    """The primary direction is confirmed by the lower timeframes."""


@dataclass(frozen=True)
class TimeframeConfirmation:
    """Result of :func:`confirm_timeframes`.

    Timeframe tuples are ordered coarsest first (``TIMEFRAMES`` order).

    Attributes:
        direction: The primary timeframe's bias (NEUTRAL if none is available).
        agreement: Confidence-weighted share (0-1) of the total configured weight
            that agrees with ``direction``.
        confirmed: Whether the direction is confirmed (always False for NEUTRAL).
        primary_timeframe: Highest available timeframe, or None if none is available.
        aligned_timeframes: Available timeframes whose bias equals ``direction``
            (includes the primary).
        conflicting_timeframes: Timeframes with the opposite bias; for a NEUTRAL
            direction, every directional timeframe.
        unavailable_timeframes: Configured timeframes that are missing or UNKNOWN.
        risk_timeframes: Available timeframes in an elevated-risk regime
            (``RISK_REGIMES``, i.e. High Volatility).
        confidence: Weight-averaged regime confidence (0-1) of the aligned
            timeframes; 0.0 if none.
        threshold: The agreement threshold that was applied.
        reason: Why the signal is (or is not) confirmed.
    """

    direction: DirectionalBias
    agreement: float
    confirmed: bool
    primary_timeframe: str | None
    aligned_timeframes: tuple[str, ...]
    conflicting_timeframes: tuple[str, ...]
    unavailable_timeframes: tuple[str, ...]
    risk_timeframes: tuple[str, ...]
    confidence: float
    threshold: float
    reason: ConfirmationReason

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict (enums as their string values, tuples as lists)."""
        return {
            "direction": self.direction.value,
            "agreement": self.agreement,
            "confirmed": self.confirmed,
            "primary_timeframe": self.primary_timeframe,
            "aligned_timeframes": list(self.aligned_timeframes),
            "conflicting_timeframes": list(self.conflicting_timeframes),
            "unavailable_timeframes": list(self.unavailable_timeframes),
            "risk_timeframes": list(self.risk_timeframes),
            "confidence": self.confidence,
            "threshold": self.threshold,
            "reason": self.reason.value,
        }


def _validated_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Return the positive weights in ``TIMEFRAMES`` order.

    Raises:
        ValueError: On an unknown timeframe, a negative or non-finite weight, or
            if no weight is positive.
    """
    unknown = sorted(set(weights) - set(TIMEFRAMES))
    if unknown:
        raise ValueError(f"Unknown timeframe(s) in weights: {unknown}; expected {list(TIMEFRAMES)}")
    for tf, w in weights.items():
        if not isinstance(w, int | float) or not math.isfinite(w) or w < 0:
            raise ValueError(f"Weight for {tf} must be a finite number >= 0, got {w!r}")
    active = {tf: float(weights[tf]) for tf in TIMEFRAMES if weights.get(tf, 0) > 0}
    if not active:
        raise ValueError("At least one timeframe weight must be positive")
    return active


def _clamped_confidence(value: float) -> float:
    """Clamp a regime confidence to [0, 1]; non-finite values count as 0."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(conf):
        return 0.0
    return min(max(conf, 0.0), 1.0)


def confirm_timeframes(
    analyses: Mapping[str, RegimeAnalysis],
    *,
    weights: Mapping[str, float] = CONFIRMATION_WEIGHTS,
    threshold: float = CONFIRMATION_THRESHOLD,
    bias_map: Mapping[MarketRegime, DirectionalBias] = REGIME_BIAS,
) -> TimeframeConfirmation:
    """Compute the multi-timeframe confirmation signal.

    Pure function: no I/O and no analyzer construction; deterministic for a given
    input. See the module docstring for the full rules.

    Args:
        analyses: ``timeframe -> RegimeAnalysis`` for any subset of the configured
            timeframes. Analyses for zero-weight timeframes are ignored.
        weights: ``timeframe -> weight`` (non-negative; keys from ``TIMEFRAMES``).
            The configured timeframes are those with a positive weight; their
            total normalizes the agreement score. Defaults to
            :data:`~mra_lib.config.timeframes.CONFIRMATION_WEIGHTS`.
        threshold: Minimum agreement in ``(0, 1]`` for a confirmation. Defaults to
            :data:`~mra_lib.config.timeframes.CONFIRMATION_THRESHOLD`.
        bias_map: ``regime -> bias``; regimes missing from it are NEUTRAL. Defaults
            to :data:`~mra_lib.config.regime_tables.REGIME_BIAS`.

    Returns:
        The confirmation result.

    Raises:
        ValueError: If ``weights`` or ``threshold`` is invalid, or ``analyses``
            contains a timeframe that has no entry in ``weights``.
    """
    active = _validated_weights(weights)
    if (
        not isinstance(threshold, int | float)
        or not math.isfinite(threshold)
        or not 0 < threshold <= 1
    ):
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    stray = sorted(set(analyses) - set(weights))
    if stray:
        raise ValueError(f"No weight configured for timeframe(s): {stray}")

    total_weight = sum(active.values())
    available = {
        tf: analyses[tf]
        for tf in active
        if tf in analyses and analyses[tf].current_regime is not MarketRegime.UNKNOWN
    }
    unavailable = tuple(tf for tf in active if tf not in available)
    bias = {
        tf: bias_map.get(a.current_regime, DirectionalBias.NEUTRAL) for tf, a in available.items()
    }
    conf = {tf: _clamped_confidence(a.regime_confidence) for tf, a in available.items()}
    risk = tuple(tf for tf, a in available.items() if a.current_regime in RISK_REGIMES)

    primary = next(iter(available), None)
    direction = bias[primary] if primary is not None else DirectionalBias.NEUTRAL

    aligned = tuple(tf for tf in available if bias[tf] is direction)
    if direction is DirectionalBias.NEUTRAL:
        conflicting = tuple(tf for tf in available if bias[tf] is not DirectionalBias.NEUTRAL)
    else:
        conflicting = tuple(tf for tf in available if bias[tf] is _OPPOSITE[direction])

    support = sum(active[tf] * conf[tf] for tf in aligned)
    aligned_weight = sum(active[tf] for tf in aligned)
    agreement = round(support / total_weight, _PRECISION)
    confidence = round(support / aligned_weight, _PRECISION) if aligned_weight > 0 else 0.0

    if primary is None:
        reason = ConfirmationReason.NO_TIMEFRAMES
    elif len(available) < MIN_CONFIRMATION_TIMEFRAMES:
        reason = ConfirmationReason.INSUFFICIENT_TIMEFRAMES
    elif direction is DirectionalBias.NEUTRAL:
        reason = ConfirmationReason.PRIMARY_NEUTRAL
    elif len(aligned) < MIN_CONFIRMATION_TIMEFRAMES:
        reason = ConfirmationReason.NO_ALIGNMENT
    elif agreement < round(float(threshold), _PRECISION):
        reason = ConfirmationReason.BELOW_THRESHOLD
    else:
        reason = ConfirmationReason.CONFIRMED

    return TimeframeConfirmation(
        direction=direction,
        agreement=agreement,
        confirmed=reason is ConfirmationReason.CONFIRMED,
        primary_timeframe=primary,
        aligned_timeframes=aligned,
        conflicting_timeframes=conflicting,
        unavailable_timeframes=unavailable,
        risk_timeframes=risk,
        confidence=confidence,
        threshold=float(threshold),
        reason=reason,
    )


def confirm_analyzer_timeframes(
    analyzer: MarketRegimeAnalyzer,
    timeframes: Iterable[str] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, RegimeAnalysis], TimeframeConfirmation]:
    """Analyze an analyzer's loaded timeframes and compute their confirmation.

    Convenience wrapper: it runs :meth:`MarketRegimeAnalyzer.analyze_current_regime`
    on already-loaded data (no data is fetched and no analyzer is built). A
    timeframe that is not loaded or cannot be analyzed (``ValueError``, including
    ``ModelNotFittedError``) is left out and counts as unavailable.

    Args:
        analyzer: An initialized analyzer.
        timeframes: Timeframes to analyze; defaults to every loaded timeframe that
            has a weight (``TIMEFRAMES`` order).
        **kwargs: Passed to :func:`confirm_timeframes` (``weights``, ``threshold``,
            ``bias_map``).

    Returns:
        ``(analyses, confirmation)``: the per-timeframe analyses that succeeded and
        the confirmation computed from them.
    """
    weights: Mapping[str, float] = kwargs.get("weights", CONFIRMATION_WEIGHTS)
    if timeframes is None:
        timeframes = [tf for tf in TIMEFRAMES if tf in analyzer.data and tf in weights]
    analyses: dict[str, RegimeAnalysis] = {}
    for tf in timeframes:
        try:
            analyses[tf] = analyzer.analyze_current_regime(tf)
        except ValueError:
            continue
    return analyses, confirm_timeframes(analyses, **kwargs)


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _tf_list(timeframes: tuple[str, ...]) -> str:
    return ", ".join(timeframes) if timeframes else "none"


def format_confirmation_report(
    confirmation: TimeframeConfirmation,
    symbol: str | None = None,
    analyses: Mapping[str, RegimeAnalysis] | None = None,
    bias_map: Mapping[MarketRegime, DirectionalBias] = REGIME_BIAS,
) -> str:
    """Format a human-readable multi-timeframe confirmation report.

    Matches the layout of
    :meth:`MarketRegimeAnalyzer.format_analysis_report`.

    Args:
        confirmation: The signal to report.
        symbol: Symbol shown in the header (omitted if None).
        analyses: Optional per-timeframe analyses, listed in ``TIMEFRAMES`` order.
        bias_map: Bias map used to label each listed timeframe.

    Returns:
        Multi-line report text (starts with a blank line, no trailing newline).
    """
    title = "MULTI-TIMEFRAME CONFIRMATION"
    if symbol:
        title += f" - {symbol}"
    status = "YES" if confirmation.confirmed else "NO"
    lines = [
        "",
        "=" * 80,
        title,
        "=" * 80,
        "🧭 SIGNAL:",
        f"   Direction: {confirmation.direction.value.capitalize()}",
        f"   Confirmed: {status} ({confirmation.reason.value})",
        f"   Agreement: {_pct(confirmation.agreement)} (threshold {_pct(confirmation.threshold)})",
        f"   Confidence: {_pct(confirmation.confidence)}",
        f"   Primary Timeframe: {confirmation.primary_timeframe or 'none'}",
        f"   Aligned: {_tf_list(confirmation.aligned_timeframes)}",
        f"   Conflicting: {_tf_list(confirmation.conflicting_timeframes)}",
        f"   Unavailable: {_tf_list(confirmation.unavailable_timeframes)}",
    ]
    if confirmation.risk_timeframes:
        lines.append(f"   ⚠️  Elevated volatility: {_tf_list(confirmation.risk_timeframes)}")

    if analyses:
        lines += ["", "📊 TIMEFRAMES:"]
        ordered = [tf for tf in TIMEFRAMES if tf in analyses]
        ordered += sorted(tf for tf in analyses if tf not in TIMEFRAMES)
        for tf in ordered:
            analysis = analyses[tf]
            regime = analysis.current_regime
            label = (
                "unavailable"
                if regime is MarketRegime.UNKNOWN
                else bias_map.get(regime, DirectionalBias.NEUTRAL).value
            )
            lines.append(
                f"   {tf}: {regime.value} ({label}, "
                f"confidence {_pct(_clamped_confidence(analysis.regime_confidence))})"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


__all__ = [
    "ConfirmationReason",
    "TimeframeConfirmation",
    "confirm_analyzer_timeframes",
    "confirm_timeframes",
    "format_confirmation_report",
]
