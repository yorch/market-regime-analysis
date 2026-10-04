"""Tests for the multi-timeframe confirmation signal (mra_lib.signals.confirmation)."""

import dataclasses
import json
import math

import pytest

from mra_lib import (
    ConfirmationReason,
    DirectionalBias,
    MarketRegime,
    TimeframeConfirmation,
    confirm_timeframes,
    format_confirmation_report,
)
from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.enums import TradingStrategy
from mra_lib.config.regime_tables import REGIME_BIAS, RISK_REGIMES, get_regime_bias
from mra_lib.config.timeframes import (
    CONFIRMATION_THRESHOLD,
    CONFIRMATION_WEIGHTS,
    TIMEFRAMES,
)
from mra_lib.signals.confirmation import confirm_analyzer_timeframes

BULL = MarketRegime.BULL_TRENDING
BEAR = MarketRegime.BEAR_TRENDING
MEAN = MarketRegime.MEAN_REVERTING
HIGH = MarketRegime.HIGH_VOLATILITY
LOW = MarketRegime.LOW_VOLATILITY
BREAK = MarketRegime.BREAKOUT
UNK = MarketRegime.UNKNOWN


def _analysis(regime: MarketRegime, confidence: float = 1.0) -> RegimeAnalysis:
    return RegimeAnalysis(
        current_regime=regime,
        hmm_state=0,
        transition_probability=0.5,
        regime_persistence=0.5,
        recommended_strategy=TradingStrategy.AVOID,
        position_sizing_multiplier=0.0,
        risk_level="Medium",
        arbitrage_opportunities=[],
        statistical_signals=[],
        key_levels={},
        regime_confidence=confidence,
    )


def _stack(**regimes: MarketRegime | tuple[MarketRegime, float]) -> dict[str, RegimeAnalysis]:
    """Build analyses from keyword args d=1D, h=1H, m=15m."""
    names = {"d": "1D", "h": "1H", "m": "15m"}
    out = {}
    for key, value in regimes.items():
        regime, conf = value if isinstance(value, tuple) else (value, 1.0)
        out[names[key]] = _analysis(regime, conf)
    return out


# ── bias table ──


def test_regime_bias_covers_every_regime():
    assert set(REGIME_BIAS) == set(MarketRegime)


@pytest.mark.parametrize(
    ("regime", "bias"),
    [
        (BULL, DirectionalBias.BULLISH),
        (BEAR, DirectionalBias.BEARISH),
        (BREAK, DirectionalBias.BULLISH),
        (MEAN, DirectionalBias.NEUTRAL),
        (LOW, DirectionalBias.NEUTRAL),
        (HIGH, DirectionalBias.NEUTRAL),
        (UNK, DirectionalBias.NEUTRAL),
    ],
)
def test_regime_bias_mapping(regime, bias):
    assert get_regime_bias(regime) is bias


def test_risk_regimes_is_high_volatility():
    assert frozenset({HIGH}) == RISK_REGIMES


def test_default_weights_follow_timeframe_order():
    assert tuple(CONFIRMATION_WEIGHTS) == TIMEFRAMES
    values = list(CONFIRMATION_WEIGHTS.values())
    assert values == sorted(values, reverse=True)


# ── confirmation rules ──


def test_confirmation_all_aligned_bullish():
    c = confirm_timeframes(_stack(d=BULL, h=BULL, m=BREAK))
    assert c.confirmed is True
    assert c.reason is ConfirmationReason.CONFIRMED
    assert c.direction is DirectionalBias.BULLISH
    assert c.agreement == pytest.approx(1.0)
    assert c.confidence == pytest.approx(1.0)
    assert c.primary_timeframe == "1D"
    assert c.aligned_timeframes == ("1D", "1H", "15m")
    assert c.conflicting_timeframes == ()
    assert c.unavailable_timeframes == ()


def test_confirmation_all_aligned_bearish():
    c = confirm_timeframes(_stack(d=(BEAR, 0.9), h=(BEAR, 0.8), m=(BEAR, 0.7)))
    assert c.confirmed is True
    assert c.direction is DirectionalBias.BEARISH
    # (0.5*0.9 + 0.3*0.8 + 0.2*0.7) / 1.0
    assert c.agreement == pytest.approx(0.83)
    assert c.confidence == pytest.approx(0.83)


def test_confirmation_primary_neutral_is_not_confirmed():
    c = confirm_timeframes(_stack(d=MEAN, h=BULL, m=BULL))
    assert c.confirmed is False
    assert c.reason is ConfirmationReason.PRIMARY_NEUTRAL
    assert c.direction is DirectionalBias.NEUTRAL
    assert c.aligned_timeframes == ("1D",)
    assert c.conflicting_timeframes == ("1H", "15m")
    assert c.agreement == pytest.approx(0.5)


def test_confirmation_high_volatility_primary_is_neutral_with_risk_flag():
    c = confirm_timeframes(_stack(d=HIGH, h=BEAR, m=BEAR))
    assert c.direction is DirectionalBias.NEUTRAL
    assert c.reason is ConfirmationReason.PRIMARY_NEUTRAL
    assert c.risk_timeframes == ("1D",)


def test_confirmation_conflicting_lower_timeframes():
    c = confirm_timeframes(_stack(d=BULL, h=BEAR, m=BEAR))
    assert c.confirmed is False
    assert c.reason is ConfirmationReason.NO_ALIGNMENT
    assert c.direction is DirectionalBias.BULLISH
    assert c.aligned_timeframes == ("1D",)
    assert c.conflicting_timeframes == ("1H", "15m")
    assert c.agreement == pytest.approx(0.5)


def test_confirmation_with_one_conflict_can_still_confirm():
    c = confirm_timeframes(_stack(d=BULL, h=BULL, m=BEAR))
    assert c.confirmed is True
    assert c.agreement == pytest.approx(0.8)
    assert c.conflicting_timeframes == ("15m",)


def test_confirmation_neutral_lower_timeframe_is_neither_aligned_nor_conflicting():
    c = confirm_timeframes(_stack(d=BULL, h=LOW, m=BULL))
    assert c.aligned_timeframes == ("1D", "15m")
    assert c.conflicting_timeframes == ()
    assert c.agreement == pytest.approx(0.7)
    assert c.confirmed is True


def test_confirmation_missing_timeframe_reduces_agreement():
    full = confirm_timeframes(_stack(d=BULL, h=BULL, m=BULL))
    missing = confirm_timeframes(_stack(d=BULL, m=BULL))
    assert missing.unavailable_timeframes == ("1H",)
    assert missing.agreement == pytest.approx(0.7)
    assert missing.agreement < full.agreement
    assert missing.confirmed is True


def test_confirmation_missing_primary_uses_next_highest():
    c = confirm_timeframes(_stack(h=BULL, m=BULL))
    assert c.primary_timeframe == "1H"
    assert c.unavailable_timeframes == ("1D",)
    assert c.agreement == pytest.approx(0.5)
    assert c.reason is ConfirmationReason.BELOW_THRESHOLD
    assert c.confirmed is False


def test_confirmation_unknown_is_treated_as_unavailable():
    c = confirm_timeframes(_stack(d=UNK, h=BULL, m=BULL))
    assert c.primary_timeframe == "1H"
    assert c.unavailable_timeframes == ("1D",)
    assert "1D" not in c.conflicting_timeframes
    assert c.agreement == pytest.approx(0.5)
    assert c.confirmed is False


def test_confirmation_all_unknown():
    c = confirm_timeframes(_stack(d=UNK, h=UNK, m=UNK))
    assert c.reason is ConfirmationReason.NO_TIMEFRAMES
    assert c.primary_timeframe is None
    assert c.direction is DirectionalBias.NEUTRAL
    assert c.agreement == 0.0
    assert c.confidence == 0.0
    assert c.unavailable_timeframes == TIMEFRAMES


def test_confirmation_empty_input():
    c = confirm_timeframes({})
    assert c.reason is ConfirmationReason.NO_TIMEFRAMES
    assert c.confirmed is False


@pytest.mark.parametrize("tf", TIMEFRAMES)
def test_confirmation_single_timeframe_is_insufficient(tf):
    c = confirm_timeframes({tf: _analysis(BULL)})
    assert c.confirmed is False
    assert c.reason is ConfirmationReason.INSUFFICIENT_TIMEFRAMES
    assert c.primary_timeframe == tf
    assert c.direction is DirectionalBias.BULLISH


def test_confirmation_single_classified_plus_unknown_is_insufficient():
    # Even a weight map that would let 1D clear the threshold alone cannot confirm
    c = confirm_timeframes(_stack(d=BULL, h=UNK), weights={"1D": 0.9, "1H": 0.1}, threshold=0.5)
    assert c.reason is ConfirmationReason.INSUFFICIENT_TIMEFRAMES
    assert c.confirmed is False


def test_confirmation_threshold_boundary_with_confidence():
    c = confirm_timeframes(_stack(d=(BULL, 0.6), h=(BULL, 0.6), m=(BULL, 0.6)))
    assert c.agreement == pytest.approx(0.6)
    assert c.confirmed is True  # exactly at threshold confirms
    c = confirm_timeframes(_stack(d=(BULL, 0.5), h=(BULL, 0.5), m=(BULL, 0.5)))
    assert c.reason is ConfirmationReason.BELOW_THRESHOLD


def test_confirmation_threshold_is_inclusive_despite_float_noise():
    # 0.1 + 0.2 != 0.3 in binary floating point
    c = confirm_timeframes(
        _stack(d=BULL, h=BULL, m=BEAR), weights={"1D": 0.1, "1H": 0.2, "15m": 0.7}, threshold=0.3
    )
    assert c.agreement == pytest.approx(0.3)
    assert c.confirmed is True


@pytest.mark.parametrize("confidence", [math.nan, math.inf, -1.0, 2.0])
def test_confirmation_tolerates_bad_confidence(confidence):
    c = confirm_timeframes(_stack(d=(BULL, confidence), h=BULL, m=BULL))
    assert 0.0 <= c.agreement <= 1.0
    assert 0.0 <= c.confidence <= 1.0
    assert math.isfinite(c.agreement)


def test_confirmation_custom_weights_and_threshold():
    stack = _stack(d=BULL, h=BULL, m=BEAR)
    assert confirm_timeframes(stack, threshold=0.81).reason is ConfirmationReason.BELOW_THRESHOLD
    assert confirm_timeframes(stack, threshold=1.0).confirmed is False
    equal = confirm_timeframes(stack, weights={"1D": 1, "1H": 1, "15m": 1})
    assert equal.agreement == pytest.approx(2 / 3)


def test_confirmation_weights_need_not_sum_to_one():
    a = confirm_timeframes(_stack(d=BULL, h=BULL, m=MEAN))
    b = confirm_timeframes(_stack(d=BULL, h=BULL, m=MEAN), weights={"1D": 5, "1H": 3, "15m": 2})
    assert a.agreement == pytest.approx(b.agreement)


def test_confirmation_zero_weight_disables_timeframe():
    c = confirm_timeframes(_stack(d=BEAR, h=BULL, m=BULL), weights={"1D": 0, "1H": 0.6, "15m": 0.4})
    assert c.primary_timeframe == "1H"
    assert c.direction is DirectionalBias.BULLISH
    assert c.agreement == pytest.approx(1.0)
    assert "1D" not in c.unavailable_timeframes
    assert c.confirmed is True


def test_confirmation_subset_weights():
    c = confirm_timeframes(_stack(h=BULL, m=BULL), weights={"1H": 0.3, "15m": 0.2})
    assert c.unavailable_timeframes == ()
    assert c.agreement == pytest.approx(1.0)
    assert c.confirmed is True


@pytest.mark.parametrize(
    "weights",
    [
        {"1D": -0.1, "1H": 0.5},
        {"1D": math.nan},
        {"1D": math.inf},
        {"1D": 0, "1H": 0},
        {},
        {"1W": 1.0},
        {"1D": "0.5"},
        {"1D": True, "1H": 0.5},
    ],
)
def test_confirmation_invalid_weights_raise(weights):
    with pytest.raises(ValueError):
        confirm_timeframes(_stack(d=BULL, h=BULL), weights=weights)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01, math.nan, "0.6", True])
def test_confirmation_invalid_threshold_raises(threshold):
    with pytest.raises(ValueError, match="threshold"):
        confirm_timeframes(_stack(d=BULL, h=BULL), threshold=threshold)


def test_confirmation_unweighted_timeframe_raises():
    with pytest.raises(ValueError, match="No weight"):
        confirm_timeframes(_stack(d=BULL, h=BULL, m=BULL), weights={"1D": 0.5, "1H": 0.3})


def test_confirmation_is_deterministic_and_input_order_independent():
    stack = _stack(d=(BULL, 0.7), h=(BEAR, 0.9), m=(BREAK, 0.4))
    reversed_stack = dict(reversed(list(stack.items())))
    results = {confirm_timeframes(stack) for _ in range(5)}
    results.add(confirm_timeframes(reversed_stack))
    assert len(results) == 1  # frozen dataclass is hashable and equal


def test_confirmation_is_frozen_and_json_safe():
    c = confirm_timeframes(_stack(d=BULL, h=BULL, m=HIGH))
    assert isinstance(c, TimeframeConfirmation)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.confirmed = False  # type: ignore[misc]
    data = c.to_dict()
    assert json.loads(json.dumps(data, allow_nan=False)) == data
    assert data["direction"] == "bullish"
    assert data["reason"] == "confirmed"
    assert data["risk_timeframes"] == ["15m"]
    assert set(data) == {f.name for f in dataclasses.fields(c)}


def test_confirmation_breakout_primary_is_bullish():
    c = confirm_timeframes(_stack(d=BREAK, h=BULL, m=MEAN))
    assert c.direction is DirectionalBias.BULLISH
    assert c.aligned_timeframes == ("1D", "1H")
    assert c.confirmed is True


def test_confirmation_subset_weights_renormalize():
    # Documented caveat: a subset of the weights renormalizes, so the default
    # "1D alone never confirms" guarantee only holds for the full default set.
    c = confirm_timeframes(_stack(d=BULL, h=(BULL, 0.0)), weights={"1D": 0.5, "1H": 0.3})
    assert c.agreement == pytest.approx(0.625)
    full = confirm_timeframes(_stack(d=BULL, h=(BULL, 0.0)))
    assert full.agreement == pytest.approx(0.5)
    assert full.confirmed is False


def test_confirmation_neutral_direction_agreement_is_neutral_consensus():
    c = confirm_timeframes(_stack(d=MEAN, h=BEAR, m=LOW))
    assert c.agreement == pytest.approx(0.7)
    assert c.confirmed is False
    assert c.conflicting_timeframes == ("1H",)


def test_confirmation_custom_bias_map():
    bias_map = {**REGIME_BIAS, MarketRegime.MEAN_REVERTING: DirectionalBias.BEARISH}
    c = confirm_timeframes(_stack(d=MEAN, h=BEAR), bias_map=bias_map)
    assert c.direction is DirectionalBias.BEARISH
    assert c.confirmed is True


def test_confirmation_default_threshold_value():
    assert confirm_timeframes(_stack(d=BULL, h=BULL)).threshold == CONFIRMATION_THRESHOLD


# ── formatting ──


def test_format_confirmation_report():
    stack = _stack(d=BULL, h=UNK, m=HIGH)
    text = format_confirmation_report(confirm_timeframes(stack), "SPY", stack)
    assert text.startswith("\n" + "=" * 80)
    assert not text.endswith("\n")
    assert "MULTI-TIMEFRAME CONFIRMATION - SPY" in text
    assert "Direction: Bullish" in text
    assert "Confirmed: NO (no_alignment)" in text
    assert "Unavailable: 1H" in text
    assert "Elevated volatility: 15m" in text
    assert "1H: Unknown (unavailable" in text
    assert "1D: Bull Trending (bullish, confidence 100.0%)" in text


def test_format_confirmation_report_without_analyses():
    text = format_confirmation_report(confirm_timeframes({}))
    assert "MULTI-TIMEFRAME CONFIRMATION\n" in text
    assert "Primary Timeframe: none" in text
    assert "TIMEFRAMES:" not in text


# ── analyzer convenience wrapper ──


class _FakeAnalyzer:
    def __init__(self, results):
        self.data = dict.fromkeys(results)
        self._results = results

    def analyze_current_regime(self, timeframe):
        result = self._results[timeframe]
        if isinstance(result, Exception):
            raise result
        return result


def test_confirm_analyzer_timeframes_skips_failures():
    fake = _FakeAnalyzer(
        {"1D": _analysis(BULL), "1H": ValueError("not fitted"), "15m": _analysis(BULL)}
    )
    analyses, c = confirm_analyzer_timeframes(fake)  # type: ignore[arg-type]
    assert set(analyses) == {"1D", "15m"}
    assert c.unavailable_timeframes == ("1H",)
    assert c.confirmed is True


def test_confirm_analyzer_timeframes_with_mock_provider():
    from mra_lib import MarketRegimeAnalyzer

    analyzer = MarketRegimeAnalyzer("SPY", periods={"1D": "2y", "1H": "6mo"}, provider_flag="mock")
    analyses, c = confirm_analyzer_timeframes(analyzer)
    assert set(analyses) <= {"1D", "1H"}
    assert c.unavailable_timeframes[-1] == "15m"  # not loaded -> unavailable
    assert c == confirm_timeframes(analyses)
