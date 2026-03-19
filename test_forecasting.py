"""Tests for TrueHMMDetector forecasting and regime stability methods."""

import numpy as np
import pandas as pd
import pytest

from market_regime_analysis.enums import MarketRegime
from market_regime_analysis.true_hmm_detector import TrueHMMDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with enough bars for HMM fitting."""
    rng = np.random.RandomState(seed)
    # Trending up then down to create distinct regimes
    trend_up = np.cumsum(rng.normal(0.001, 0.01, n // 2))
    trend_down = np.cumsum(rng.normal(-0.001, 0.015, n - n // 2))
    log_prices = np.concatenate([trend_up, trend_down])
    close = 100 * np.exp(log_prices)

    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Volume": rng.randint(500_000, 2_000_000, n),
        },
        index=dates,
    )


def _fitted_hmm(n_states: int = 4, n: int = 300) -> tuple[TrueHMMDetector, pd.DataFrame]:
    """Return a fitted TrueHMMDetector and the data used to fit it."""
    df = _make_synthetic_ohlcv(n)
    hmm = TrueHMMDetector(n_states=n_states, n_iter=50)
    hmm.fit(df)
    return hmm, df


# ---------------------------------------------------------------------------
# Tests: _map_state_index_to_regime
# ---------------------------------------------------------------------------


class TestMapStateIndexToRegime:
    """Tests for the state-to-regime mapping helper."""

    def test_returns_valid_regime_for_all_states(self):
        hmm, _ = _fitted_hmm()
        for i in range(hmm.n_states):
            regime = hmm._map_state_index_to_regime(i)
            assert isinstance(regime, MarketRegime)

    def test_out_of_range_returns_unknown(self):
        hmm, _ = _fitted_hmm()
        assert hmm._map_state_index_to_regime(999) == MarketRegime.UNKNOWN


class TestGetStateRegimeMap:
    """Tests for get_state_regime_map."""

    def test_returns_all_states(self):
        hmm, _ = _fitted_hmm()
        mapping = hmm.get_state_regime_map()
        assert len(mapping) == hmm.n_states
        assert set(mapping.keys()) == set(range(hmm.n_states))

    def test_raises_if_not_fitted(self):
        hmm = TrueHMMDetector()
        with pytest.raises(ValueError, match="fitted"):
            hmm.get_state_regime_map()


# ---------------------------------------------------------------------------
# Tests: forecast_regime_probabilities
# ---------------------------------------------------------------------------


class TestForecastRegimeProbabilities:
    """Tests for single-step forecasting."""

    def test_returns_correct_shape(self):
        hmm, df = _fitted_hmm()
        probs = hmm.forecast_regime_probabilities(df, n_steps=1)
        assert probs.shape == (hmm.n_states,)

    def test_probabilities_sum_to_one(self):
        hmm, df = _fitted_hmm()
        probs = hmm.forecast_regime_probabilities(df, n_steps=1)
        assert np.isclose(probs.sum(), 1.0, atol=1e-6)

    def test_n_step_probabilities_sum_to_one(self):
        hmm, df = _fitted_hmm()
        for n in [1, 3, 10, 50]:
            probs = hmm.forecast_regime_probabilities(df, n_steps=n)
            assert np.isclose(probs.sum(), 1.0, atol=1e-6), f"Failed for n_steps={n}"

    def test_one_step_matches_manual_computation(self):
        hmm, df = _fitted_hmm()
        # Get current posterior manually
        X = hmm._prepare_features(df)
        X_scaled = hmm.scaler.transform(X)
        pi_t = hmm.model.predict_proba(X_scaled)[-1]
        expected = pi_t @ hmm.transition_matrix
        actual = hmm.forecast_regime_probabilities(df, n_steps=1)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_large_n_converges_to_stationary(self):
        hmm, df = _fitted_hmm()
        probs_100 = hmm.forecast_regime_probabilities(df, n_steps=100)
        probs_200 = hmm.forecast_regime_probabilities(df, n_steps=200)
        # Should converge — difference should be small
        np.testing.assert_allclose(probs_100, probs_200, atol=0.05)

    def test_raises_if_not_fitted(self):
        hmm = TrueHMMDetector()
        df = _make_synthetic_ohlcv(100)
        with pytest.raises(ValueError, match="fitted"):
            hmm.forecast_regime_probabilities(df)

    def test_raises_if_n_steps_zero(self):
        hmm, df = _fitted_hmm()
        with pytest.raises(ValueError, match="n_steps"):
            hmm.forecast_regime_probabilities(df, n_steps=0)

    def test_all_probabilities_non_negative(self):
        hmm, df = _fitted_hmm()
        probs = hmm.forecast_regime_probabilities(df, n_steps=5)
        assert np.all(probs >= -1e-10)


# ---------------------------------------------------------------------------
# Tests: forecast_regime_sequence
# ---------------------------------------------------------------------------


class TestForecastRegimeSequence:
    """Tests for multi-step sequence forecasting."""

    def test_returns_correct_length(self):
        hmm, df = _fitted_hmm()
        seq = hmm.forecast_regime_sequence(df, n_steps=5)
        assert len(seq) == 5

    def test_steps_are_sequential(self):
        hmm, df = _fitted_hmm()
        seq = hmm.forecast_regime_sequence(df, n_steps=5)
        assert [f["step"] for f in seq] == [1, 2, 3, 4, 5]

    def test_regime_probabilities_sum_to_one(self):
        hmm, df = _fitted_hmm()
        seq = hmm.forecast_regime_sequence(df, n_steps=3)
        for f in seq:
            total = sum(f["regime_probabilities"].values())
            assert np.isclose(total, 1.0, atol=1e-6)

    def test_most_likely_regime_is_valid(self):
        hmm, df = _fitted_hmm()
        seq = hmm.forecast_regime_sequence(df, n_steps=3)
        for f in seq:
            assert isinstance(f["most_likely_regime"], MarketRegime)
            assert f["most_likely_regime_probability"] > 0

    def test_most_likely_matches_max_probability(self):
        hmm, df = _fitted_hmm()
        seq = hmm.forecast_regime_sequence(df, n_steps=3)
        for f in seq:
            rp = f["regime_probabilities"]
            expected_regime = max(rp, key=rp.get)
            assert f["most_likely_regime"] == expected_regime

    def test_state_probabilities_match_regime_aggregation(self):
        """State probs and regime probs should be consistent."""
        hmm, df = _fitted_hmm()
        seq = hmm.forecast_regime_sequence(df, n_steps=2)
        state_map = hmm.get_state_regime_map()

        for f in seq:
            # Manually aggregate state probs by regime
            expected = {}
            for i, p in enumerate(f["state_probabilities"]):
                r = state_map[i]
                expected[r] = expected.get(r, 0.0) + p

            for regime, prob in expected.items():
                assert np.isclose(f["regime_probabilities"][regime], prob, atol=1e-10)

    def test_raises_if_not_fitted(self):
        hmm = TrueHMMDetector()
        df = _make_synthetic_ohlcv(100)
        with pytest.raises(ValueError, match="fitted"):
            hmm.forecast_regime_sequence(df)


# ---------------------------------------------------------------------------
# Tests: get_regime_stability
# ---------------------------------------------------------------------------


class TestGetRegimeStability:
    """Tests for regime stability metrics."""

    def test_has_expected_keys(self):
        hmm, _ = _fitted_hmm()
        stab = hmm.get_regime_stability()
        assert "self_transition_probs" in stab
        assert "expected_durations" in stab
        assert "stationary_distribution" in stab
        assert "stationary_regimes" in stab

    def test_self_transition_probs_in_range(self):
        hmm, _ = _fitted_hmm()
        stab = hmm.get_regime_stability()
        for i, p in stab["self_transition_probs"].items():
            assert 0.0 <= p <= 1.0, f"State {i}: {p}"

    def test_expected_durations_positive(self):
        hmm, _ = _fitted_hmm()
        stab = hmm.get_regime_stability()
        for i, d in stab["expected_durations"].items():
            assert d >= 1.0, f"State {i}: {d}"

    def test_stationary_distribution_sums_to_one(self):
        hmm, _ = _fitted_hmm()
        stab = hmm.get_regime_stability()
        assert np.isclose(stab["stationary_distribution"].sum(), 1.0, atol=1e-6)

    def test_stationary_distribution_non_negative(self):
        hmm, _ = _fitted_hmm()
        stab = hmm.get_regime_stability()
        assert np.all(stab["stationary_distribution"] >= -1e-10)

    def test_stationary_regimes_sum_to_one(self):
        hmm, _ = _fitted_hmm()
        stab = hmm.get_regime_stability()
        total = sum(stab["stationary_regimes"].values())
        assert np.isclose(total, 1.0, atol=1e-6)

    def test_stationary_is_fixed_point(self):
        """Stationary distribution should be unchanged by one transition step."""
        hmm, _ = _fitted_hmm()
        stab = hmm.get_regime_stability()
        pi = stab["stationary_distribution"]
        pi_next = pi @ hmm.transition_matrix
        np.testing.assert_allclose(pi, pi_next, atol=1e-6)

    def test_raises_if_not_fitted(self):
        hmm = TrueHMMDetector()
        with pytest.raises(ValueError, match="fitted"):
            hmm.get_regime_stability()
