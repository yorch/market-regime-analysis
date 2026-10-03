"""
Tests for the deprecated ``HiddenMarkovRegimeDetector`` alias and shared detector helpers.

The GMM implementation was removed (its feature/mapping internals had their
own tests here); the name now aliases :class:`TrueHMMDetector`.
"""

import warnings

import numpy as np
import pytest

from mra_lib.config.enums import MarketRegime
from mra_lib.indicators import RegimeDetector, regime_persistence, transition_probability
from mra_lib.indicators.hmm_detector import HiddenMarkovRegimeDetector
from mra_lib.indicators.true_hmm_detector import TrueHMMDetector


class TestDeprecatedAlias:
    def test_construction_warns(self):
        with pytest.warns(DeprecationWarning, match="TrueHMMDetector"):
            HiddenMarkovRegimeDetector()

    def test_is_true_hmm_with_same_defaults(self):
        with pytest.warns(DeprecationWarning):
            det = HiddenMarkovRegimeDetector(n_states=4)
        assert isinstance(det, TrueHMMDetector)
        assert det.n_states == 4
        assert det.covariance_type == "diag"

    def test_package_import_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            import mra_lib  # noqa: F401

    def test_fits_and_predicts_like_true_hmm(self, synthetic_ohlcv):
        df = synthetic_ohlcv(n=300)
        with pytest.warns(DeprecationWarning):
            alias = HiddenMarkovRegimeDetector(n_states=3, n_init=2).fit(df)
        ref = TrueHMMDetector(n_states=3, n_init=2).fit(df)
        assert alias.predict_regime(df) == ref.predict_regime(df)

    def test_compare_with_gmm_is_deprecated(self, synthetic_ohlcv):
        df = synthetic_ohlcv(n=300)
        det = TrueHMMDetector(n_states=3, n_init=2).fit(df)
        with pytest.warns(DeprecationWarning):
            out = det.compare_with_gmm(df, det)
        assert out["regime_agreement"] is True


class TestProtocol:
    def test_true_hmm_satisfies_protocol(self):
        assert isinstance(TrueHMMDetector(), RegimeDetector)


class TestRegimePersistence:
    @pytest.mark.parametrize("states", [[], [3]])
    def test_too_short_is_zero(self, states):
        assert regime_persistence(np.array(states, dtype=int)) == 0.0

    def test_all_same_state(self):
        assert regime_persistence(np.array([2] * 30)) == 1.0

    def test_alternating(self):
        assert regime_persistence(np.array([0, 1] * 10)) == pytest.approx(0.5)

    def test_lookback_respected(self):
        states = np.array([1] * 50 + [0] * 10)
        assert regime_persistence(states, lookback=10) == 1.0
        assert regime_persistence(states, lookback=20) == pytest.approx(0.5)

    def test_detector_delegates(self):
        states = np.array([1] * 5 + [0] * 15)
        assert TrueHMMDetector().calculate_regime_persistence(states) == regime_persistence(states)


class TestTransitionProbability:
    T = np.array([[0.9, 0.1], [0.4, 0.6]])

    def test_lookup(self):
        assert transition_probability(self.T, 1, 0) == pytest.approx(0.4)

    def test_unfitted_raises(self):
        with pytest.raises(ValueError, match="fitted"):
            transition_probability(None, 0, 0)
        with pytest.raises(ValueError, match="fitted"):
            TrueHMMDetector().get_transition_probability(0, 0)

    @pytest.mark.parametrize(("a", "b", "msg"), [(-1, 0, "from_state"), (0, 2, "to_state")])
    def test_out_of_range(self, a, b, msg):
        with pytest.raises(ValueError, match=msg):
            transition_probability(self.T, a, b)


def test_unknown_for_unfitted_state_lookup():
    assert TrueHMMDetector()._map_state_index_to_regime(0) == MarketRegime.UNKNOWN
