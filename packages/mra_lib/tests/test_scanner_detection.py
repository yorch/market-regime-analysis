"""Tests for the pure regime-change detection (mra_lib.scanner.detection)."""

from datetime import datetime, timedelta

import pytest

from mra_lib.config.enums import DirectionalBias
from mra_lib.errors import InvalidParametersError
from mra_lib.scanner import (
    AlertPolicy,
    ChangeKind,
    SuppressReason,
    detect_change,
    in_cooldown,
    previous_run,
)
from mra_lib.signals.confirmation import ConfirmationReason, TimeframeConfirmation
from mra_lib.storage import RegimeRecord

BULL, BEAR, MR, UNKNOWN = "Bull Trending", "Bear Trending", "Mean Reverting", "Unknown"
T0 = datetime(2026, 1, 5)


def rec(
    regime: str, day: int, *, tf: str = "1D", confidence: float = 0.9, symbol: str = "SPY"
) -> RegimeRecord:
    return RegimeRecord(
        symbol=symbol,
        timeframe=tf,
        bar_time=T0 + timedelta(days=day),
        regime=regime,
        confidence=confidence,
        persistence=0.5,
        transition_probability=0.5,
        recommended_strategy="Trend Following",
        provider="mock",
    )


def confirmation(
    direction: DirectionalBias = DirectionalBias.BULLISH, *, confirmed: bool = True
) -> TimeframeConfirmation:
    return TimeframeConfirmation(
        direction=direction,
        agreement=0.8 if confirmed else 0.3,
        confirmed=confirmed,
        primary_timeframe="1D",
        aligned_timeframes=("1D", "1H"),
        conflicting_timeframes=(),
        unavailable_timeframes=(),
        risk_timeframes=(),
        confidence=0.9,
        threshold=0.6,
        reason=ConfirmationReason.CONFIRMED if confirmed else ConfirmationReason.BELOW_THRESHOLD,
    )


NO_CONFIRM = AlertPolicy(require_confirmation=False)


class TestClassification:
    def test_first_observation_is_baseline_not_alert(self):
        d = detect_change(None, rec(BULL, 1), confirmation=confirmation())
        assert d.kind is ChangeKind.BASELINE
        assert not d.alert and not d.changed
        assert d.previous_regime is None

    def test_same_bar_rescanned_is_not_a_change_even_if_regime_differs(self):
        d = detect_change(rec(BEAR, 1), rec(BULL, 1), confirmation=confirmation())
        assert d.kind is ChangeKind.SAME_BAR
        assert not d.alert

    def test_older_bar_is_stale(self):
        d = detect_change(rec(BEAR, 2), rec(BULL, 1), policy=NO_CONFIRM)
        assert d.kind is ChangeKind.STALE_BAR
        assert not d.alert

    def test_newer_bar_same_regime_is_unchanged(self):
        d = detect_change(rec(BULL, 1), rec(BULL, 2), confirmation=confirmation())
        assert d.kind is ChangeKind.UNCHANGED
        assert not d.alert

    def test_newer_bar_changed_regime_with_confirmation_alerts(self):
        d = detect_change(rec(BEAR, 1), rec(BULL, 2), confirmation=confirmation())
        assert d.kind is ChangeKind.CHANGED
        assert d.alert and d.suppressed is None
        assert (d.previous_regime, d.new_regime) == (BEAR, BULL)

    def test_changed_without_confirmation_is_suppressed_by_default(self):
        d = detect_change(rec(BEAR, 1), rec(BULL, 2), confirmation=None)
        assert d.changed and not d.alert
        assert d.suppressed is SuppressReason.NOT_CONFIRMED

    def test_changed_with_unconfirmed_signal_is_suppressed(self):
        d = detect_change(rec(BEAR, 1), rec(BULL, 2), confirmation=confirmation(confirmed=False))
        assert d.suppressed is SuppressReason.NOT_CONFIRMED

    def test_changed_alerts_when_confirmation_not_required(self):
        d = detect_change(rec(BEAR, 1), rec(BULL, 2), policy=NO_CONFIRM, confirmation=None)
        assert d.alert

    def test_direction_mismatch_is_suppressed(self):
        d = detect_change(
            rec(BULL, 1), rec(BEAR, 2), confirmation=confirmation(DirectionalBias.BULLISH)
        )
        assert d.suppressed is SuppressReason.DIRECTION_MISMATCH

    def test_neutral_new_regime_never_matches_a_confirmation(self):
        d = detect_change(rec(BULL, 1), rec(MR, 2), confirmation=confirmation())
        assert d.suppressed is SuppressReason.DIRECTION_MISMATCH
        assert detect_change(rec(BULL, 1), rec(MR, 2), policy=NO_CONFIRM).alert

    def test_below_min_confidence_is_suppressed(self):
        d = detect_change(rec(BEAR, 1), rec(BULL, 2, confidence=0.59), confirmation=confirmation())
        assert d.suppressed is SuppressReason.LOW_CONFIDENCE
        lenient = AlertPolicy(min_confidence=0.5)
        assert detect_change(
            rec(BEAR, 1), rec(BULL, 2, confidence=0.59), policy=lenient, confirmation=confirmation()
        ).alert

    def test_unwatched_timeframe_is_suppressed(self):
        d = detect_change(rec(BEAR, 1, tf="1H"), rec(BULL, 2, tf="1H"), policy=NO_CONFIRM)
        assert d.suppressed is SuppressReason.UNWATCHED
        watched = AlertPolicy(timeframes_to_watch=("1D", "1H"), require_confirmation=False)
        assert detect_change(rec(BEAR, 1, tf="1H"), rec(BULL, 2, tf="1H"), policy=watched).alert

    def test_change_to_unknown_is_suppressed(self):
        d = detect_change(rec(BULL, 1), rec(UNKNOWN, 2), policy=NO_CONFIRM)
        assert d.suppressed is SuppressReason.UNKNOWN_REGIME

    def test_mismatched_records_raise(self):
        with pytest.raises(InvalidParametersError):
            detect_change(rec(BULL, 1, symbol="QQQ"), rec(BEAR, 2))
        with pytest.raises(InvalidParametersError):
            detect_change(rec(BULL, 1, tf="1H"), rec(BEAR, 2))


class TestCooldown:
    def test_previous_run_measures_run_and_visible_transition(self):
        history = [rec(BEAR, 1), rec(BEAR, 2), rec(BULL, 3), rec(BULL, 4)]
        assert previous_run(history[-1], history) == (2, T0 + timedelta(days=3))

    def test_previous_run_without_visible_transition(self):
        history = [rec(BULL, 1), rec(BULL, 2)]
        assert previous_run(history[-1], history) == (2, None)
        # previous is added if the history slice does not contain it
        assert previous_run(rec(BULL, 3), []) == (1, None)

    def test_previous_run_ignores_newer_and_foreign_records(self):
        prev = rec(BULL, 2)
        history = [
            rec(BEAR, 1),
            prev,
            rec(BEAR, 3),
            rec(BEAR, 1, tf="1H"),
            rec(BULL, 1, symbol="Q"),
        ]
        assert previous_run(prev, history) == (1, T0 + timedelta(days=2))

    def test_flip_back_after_one_bar_is_cooldown(self):
        # A A A B | A  -> B lasted one bar after a visible transition
        history = [rec(BEAR, 1), rec(BEAR, 2), rec(BEAR, 3), rec(BULL, 4)]
        d = detect_change(history[-1], rec(BEAR, 5), policy=NO_CONFIRM, history=history)
        assert d.suppressed is SuppressReason.COOLDOWN

    def test_change_after_long_run_alerts(self):
        history = [rec(BEAR, 1), rec(BULL, 2), rec(BULL, 3)]
        d = detect_change(history[-1], rec(BEAR, 4), policy=NO_CONFIRM, history=history)
        assert d.alert

    def test_baseline_run_is_not_in_cooldown(self):
        # Only one stored bar: the run's start is not visible, so no cooldown
        history = [rec(BEAR, 1)]
        d = detect_change(history[-1], rec(BULL, 2), policy=NO_CONFIRM, history=history)
        assert d.alert

    def test_cooldown_bars_zero_disables(self):
        policy = AlertPolicy(require_confirmation=False, cooldown_bars=0)
        history = [rec(BEAR, 1), rec(BULL, 2)]
        assert detect_change(history[-1], rec(BEAR, 3), policy=policy, history=history).alert

    def test_larger_cooldown_bars(self):
        policy = AlertPolicy(require_confirmation=False, cooldown_bars=3)
        history = [rec(BEAR, 1), rec(BULL, 2), rec(BULL, 3), rec(BULL, 4)]
        assert in_cooldown(history[-1], rec(BEAR, 5), history, policy)
        history.append(rec(BULL, 5))
        assert not in_cooldown(history[-1], rec(BEAR, 6), history, policy)
        assert policy.history_lookback == 4

    def test_duration_cooldown(self):
        policy = AlertPolicy(
            require_confirmation=False, cooldown_bars=0, cooldown=timedelta(days=5)
        )
        history = [rec(BEAR, 1), rec(BULL, 2), rec(BULL, 3)]
        # Bull started on day 2; day 4 is only 2 days later
        assert detect_change(history[-1], rec(BEAR, 4), policy=policy, history=history).suppressed
        # Day 7 is 5 days after the transition: allowed
        assert detect_change(history[-1], rec(BEAR, 7), policy=policy, history=history).alert
        assert policy.history_lookback == 1000

    def test_cooldown_checked_last(self):
        history = [rec(BEAR, 1), rec(BULL, 2)]
        d = detect_change(history[-1], rec(BEAR, 3, confidence=0.1), history=history)
        assert d.suppressed is SuppressReason.LOW_CONFIDENCE


class TestPolicyValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"timeframes_to_watch": ()},
            {"timeframes_to_watch": ("4H",)},
            {"min_confidence": 1.5},
            {"min_confidence": float("nan")},
            {"cooldown_bars": -1},
            {"cooldown": timedelta(days=-1)},
        ],
    )
    def test_invalid_policy_raises(self, kwargs):
        with pytest.raises(InvalidParametersError):
            AlertPolicy(**kwargs)

    def test_string_watch_is_one_timeframe(self):
        assert AlertPolicy(timeframes_to_watch="1H").timeframes_to_watch == ("1H",)  # type: ignore[arg-type]

    def test_defaults(self):
        policy = AlertPolicy()
        assert policy.timeframes_to_watch == ("1D",)
        assert policy.require_confirmation is True
        assert policy.min_confidence == 0.6
        assert policy.cooldown_bars == 1
        assert policy.history_lookback == 2
