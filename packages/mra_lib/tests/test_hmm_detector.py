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
    def test_not_fitted_raises(self):
        """Same edge-case contract as TrueHMMDetector."""
        hmm = HiddenMarkovRegimeDetector()
        with pytest.raises(ValueError, match="fitted"):
            hmm.get_transition_probability(0, 1)

    def test_valid_transition(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        prob = hmm.get_transition_probability(0, 1)
        assert 0 <= prob <= 1

    def test_out_of_range_raises(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        for args in [(-1, 0), (0, 10), (10, 0)]:
            with pytest.raises(ValueError, match="Invalid"):
                hmm.get_transition_probability(*args)

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


class TestRegressionFixes:
    """Regression tests for the 2026-10-03 analytics review fixes."""

    def test_transition_matrix_laplace_smoothing_unseen_row_uniform(self):
        hmm = HiddenMarkovRegimeDetector(n_states=3)
        # State 2 is never visited -> its row must be uniform, not all zeros.
        tm = hmm._estimate_transition_matrix(np.array([0, 0, 1, 1, 0, 1]))
        np.testing.assert_allclose(tm.sum(axis=1), 1.0)
        np.testing.assert_allclose(tm[2], [1 / 3, 1 / 3, 1 / 3])
        assert (tm > 0).all()

    def test_state_means_have_no_nans(self):
        hmm = HiddenMarkovRegimeDetector(n_states=6)
        hmm.fit(_make_ohlcv(300))
        assert not np.isnan(hmm.state_means).any()

    def test_predict_regime_returns_plain_python_types(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        _, state, confidence = hmm.predict_regime(df)
        assert type(state) is int
        assert type(confidence) is float

    def test_predict_with_states_matches_predict_regime(self):
        hmm = HiddenMarkovRegimeDetector(n_states=4)
        df = _make_ohlcv(300)
        hmm.fit(df)
        regime, states, conf = hmm.predict_with_states(df)
        assert (regime, int(states[-1]), conf) == hmm.predict_regime(df)
        assert len(states) == len(hmm._prepare_features(df))

    def test_fit_error_chains_cause(self):
        hmm = HiddenMarkovRegimeDetector(n_states=6)
        with pytest.raises(ValueError) as exc:
            hmm.fit(_make_ohlcv(30))
        assert exc.value.__cause__ is not None

    def _fitted_with_synthetic_X(self, rows):
        """Detector whose feature layout puts sma_9 where trend_strength used to be."""
        hmm = HiddenMarkovRegimeDetector(n_states=2)
        hmm._prepare_features(_make_ohlcv(120))  # populate feature_names
        names = hmm.feature_names
        X = np.zeros((len(rows), len(names)))
        for i, row in enumerate(rows):
            for name, value in row.items():
                X[i, names.index(name)] = value
        return hmm, X

    def test_mapping_uses_trend_strength_by_name_not_column_9(self):
        names = HiddenMarkovRegimeDetector()
        names._prepare_features(_make_ohlcv(120))
        # Column 9 is a price-level SMA, not trend strength.
        assert names.feature_names[9] != "trend_strength"

        base = {"volatility": 0.0}
        # Volatility spread so the current state sits in the middle quartiles.
        history = [{"volatility": v} for v in (-2.0, -1.0, 1.0, 2.0)] * 5
        bull_now = {**base, "returns": 1.0, "trend_strength": 1.0, "sma_9": -5.0}
        hmm, X = self._fitted_with_synthetic_X(history + [bull_now] * 20)
        states = np.array([0] * 20 + [1] * 20)
        assert hmm._map_states_to_regimes(X, states) == MarketRegime.BULL_TRENDING

        bear_now = {**base, "returns": -1.0, "trend_strength": -1.0, "sma_9": 5.0}
        hmm, X = self._fitted_with_synthetic_X(history + [bear_now] * 20)
        assert hmm._map_states_to_regimes(X, states) == MarketRegime.BEAR_TRENDING

    def test_mapping_thresholds_are_in_z_units(self):
        # Tiny trend z-score (old 0.001 threshold would call this a trend) -> not BULL.
        history = [{"volatility": v} for v in (-2.0, -1.0, 1.0, 2.0)] * 5
        weak = {"volatility": 0.0, "returns": 0.05, "trend_strength": 0.05}
        hmm, X = self._fitted_with_synthetic_X(history + [weak] * 20)
        states = np.array([0] * 20 + [1] * 20)
        assert hmm._map_states_to_regimes(X, states) != MarketRegime.BULL_TRENDING
