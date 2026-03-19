"""Tests for indicators/hmm_detector.py — HiddenMarkovRegimeDetector."""

import numpy as np
import pandas as pd
import pytest

from mra_lib.config.enums import MarketRegime
from mra_lib.indicators.hmm_detector import HiddenMarkovRegimeDetector


def _make_ohlcv(n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0.0005, 0.015, n))
    close = np.maximum(close, 10)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    low = np.maximum(low, 1)
    opn = close + rng.normal(0, 0.5, n)
    vol = rng.integers(1_000_000, 10_000_000, n)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


class TestInit:
    def test_default_states(self):
        hmm = HiddenMarkovRegimeDetector()
        assert hmm.n_states == 6
        assert hmm.fitted is False

    def test_custom_states(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        assert hmm.n_states == 4


class TestPrepareFeatures:
    def test_insufficient_data(self):
        hmm = HiddenMarkovRegimeDetector()
        df = _make_ohlcv(30)
        with pytest.raises(ValueError, match="Insufficient data"):
            hmm._prepare_features(df)

    def test_feature_names(self):
        hmm = HiddenMarkovRegimeDetector()
        df = _make_ohlcv(300)
        hmm._prepare_features(df)
        assert "returns" in hmm.feature_names
        assert "volatility" in hmm.feature_names
        assert "skewness" in hmm.feature_names
        assert "kurtosis" in hmm.feature_names
        assert "autocorr_1" in hmm.feature_names

    def test_no_nans_in_output(self):
        hmm = HiddenMarkovRegimeDetector()
        df = _make_ohlcv(300)
        features = hmm._prepare_features(df)
        assert not features.isna().any().any()

    def test_zero_volume_handled(self):
        hmm = HiddenMarkovRegimeDetector()
        df = _make_ohlcv(300)
        df["Volume"] = 0
        features = hmm._prepare_features(df)
        assert "volume_ratio" in features.columns


class TestFit:
    def test_fit_success(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        result = hmm.fit(df)
        assert result is hmm  # Method chaining
        assert hmm.fitted is True
        assert hmm.gmm is not None
        assert hmm.scaler is not None
        assert hmm.transition_matrix is not None
        assert hmm.state_means is not None

    def test_fit_insufficient_data(self):
        hmm = HiddenMarkovRegimeDetector(n_states=6)
        df = _make_ohlcv(30)
        with pytest.raises(ValueError):
            hmm.fit(df)

    def test_transition_matrix_shape(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        assert hmm.transition_matrix.shape == (4, 4)

    def test_transition_matrix_rows_sum_to_one(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        row_sums = hmm.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_state_means_shape(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        assert hmm.state_means.shape[0] == 4


class TestEstimateTransitionMatrix:
    def test_simple_sequence(self):
        hmm = HiddenMarkovRegimeDetector(n_states=3)
        states = np.array([0, 0, 1, 1, 2, 0, 0, 1])
        tm = hmm._estimate_transition_matrix(states)
        assert tm.shape == (3, 3)
        # All values should be probabilities
        assert (tm >= 0).all()
        row_sums = tm.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)


class TestPredictRegime:
    def test_not_fitted_raises(self):
        hmm = HiddenMarkovRegimeDetector()
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match="fitted"):
            hmm.predict_regime(df)

    def test_predict_returns_tuple(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        regime, state, confidence = hmm.predict_regime(df)
        assert isinstance(regime, MarketRegime)
        assert isinstance(state, (int, np.integer))
        assert 0 <= confidence <= 1.0

    def test_state_in_valid_range(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        _, state, _ = hmm.predict_regime(df)
        assert 0 <= state < 4


class TestMapStatesToRegimes:
    def test_empty_states(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        result = hmm._map_states_to_regimes(np.array([]), np.array([]))
        assert result == MarketRegime.UNKNOWN

    def test_returns_valid_regime(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        features = hmm._prepare_features(df)
        X_scaled = hmm.scaler.transform(features)
        states = hmm.gmm.predict(X_scaled)
        regime = hmm._map_states_to_regimes(X_scaled, states)
        assert regime in list(MarketRegime)


class TestGetTransitionProbability:
    def test_not_fitted_returns_zero(self):
        hmm = HiddenMarkovRegimeDetector()
        assert hmm.get_transition_probability(0, 1) == 0.0

    def test_valid_transition(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        prob = hmm.get_transition_probability(0, 1)
        assert 0 <= prob <= 1

    def test_out_of_range_returns_zero(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        assert hmm.get_transition_probability(-1, 0) == 0.0
        assert hmm.get_transition_probability(0, 10) == 0.0
        assert hmm.get_transition_probability(10, 0) == 0.0

    def test_self_transition(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        prob = hmm.get_transition_probability(0, 0)
        assert 0 <= prob <= 1


class TestCalculateRegimePersistence:
    def test_single_state(self):
        hmm = HiddenMarkovRegimeDetector()
        states = np.array([0])
        assert hmm.calculate_regime_persistence(states) == 0.0

    def test_all_same_state(self):
        hmm = HiddenMarkovRegimeDetector()
        states = np.array([2, 2, 2, 2, 2])
        assert hmm.calculate_regime_persistence(states) == 1.0

    def test_alternating_states(self):
        hmm = HiddenMarkovRegimeDetector()
        states = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        persistence = hmm.calculate_regime_persistence(states)
        assert persistence == pytest.approx(0.5)

    def test_lookback_respected(self):
        hmm = HiddenMarkovRegimeDetector()
        states = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        # With full lookback, current state (1) appears 6/10 = 60%
        p_full = hmm.calculate_regime_persistence(states, lookback=20)
        # With lookback=5, current state (1) appears 5/5 = 100%
        p_short = hmm.calculate_regime_persistence(states, lookback=5)
        assert p_short >= p_full
