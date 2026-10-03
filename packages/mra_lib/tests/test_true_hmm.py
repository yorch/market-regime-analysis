"""TrueHMMDetector (hmmlearn) on deterministic synthetic data."""

import logging

import numpy as np
import pandas as pd
import pytest

from mra_lib.config.enums import MarketRegime
from mra_lib.indicators.true_hmm_detector import TrueHMMDetector, select_n_states


@pytest.fixture
def fitted(synthetic_ohlcv):
    df = synthetic_ohlcv(n=300)
    return TrueHMMDetector(n_states=4, n_iter=50).fit(df), df


def test_true_hmm_convergence_info_after_fit(fitted):
    hmm, _ = fitted

    info = hmm.get_training_convergence()

    assert info["fitted"] is True
    assert info["n_states"] == 4
    assert info["n_features"] == len(hmm.feature_names) > 0
    assert np.isfinite(info["log_likelihood"])


def test_true_hmm_transition_matrix_is_row_stochastic(fitted):
    hmm, _ = fitted

    assert hmm.transition_matrix.shape == (4, 4)
    np.testing.assert_allclose(hmm.transition_matrix.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("use_viterbi", [True, False])
def test_true_hmm_predict_regime_returns_valid_tuple(fitted, use_viterbi):
    hmm, df = fitted

    regime, state, confidence = hmm.predict_regime(df, use_viterbi=use_viterbi)

    assert isinstance(regime, MarketRegime)
    assert 0 <= state < 4
    assert 0.0 <= confidence <= 1.0


def test_true_hmm_is_deterministic_for_fixed_seed(synthetic_ohlcv):
    df = synthetic_ohlcv(n=300)

    first = TrueHMMDetector(n_states=4, n_iter=50).fit(df).predict_regime(df)
    second = TrueHMMDetector(n_states=4, n_iter=50).fit(df).predict_regime(df)

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == pytest.approx(second[2])


def test_true_hmm_predict_before_fit_raises(synthetic_ohlcv):
    with pytest.raises(ValueError, match="fitted"):
        TrueHMMDetector(n_states=4).predict_regime(synthetic_ohlcv(n=100))


# ---------------------------------------------------------------------------
# Model unification / stability (2026-10-03 review, P1 analytics)
# ---------------------------------------------------------------------------


def _labels(det: TrueHMMDetector, df) -> list:
    return list(det.regime_history(df))


def test_defaults_are_small_regularized_model():
    det = TrueHMMDetector()
    assert det.covariance_type == "diag"
    assert det.n_init >= 5
    assert det.min_covar > 0
    assert len(det.feature_names) <= 7


def test_insufficient_data_error_mentions_parameter_count(synthetic_ohlcv):
    det = TrueHMMDetector(n_states=6)
    df = synthetic_ohlcv(n=det.min_training_bars - 1)
    with pytest.raises(ValueError, match=r"free parameters need at least"):
        det.fit(df)
    TrueHMMDetector(n_states=6, n_init=1).fit(synthetic_ohlcv(n=det.min_training_bars))


def test_states_sorted_by_volatility(fitted):
    hmm, _ = fitted
    vols = [s.rel_vol for s in hmm.state_summaries]
    assert vols == sorted(vols)


def test_canonical_order_keeps_model_consistent(fitted):
    """Permuting states must leave the likelihood and posteriors self-consistent."""
    hmm, df = fitted
    X = hmm.scaler.transform(hmm._prepare_features(df))
    np.testing.assert_allclose(hmm.model.transmat_.sum(axis=1), 1.0)
    assert hmm.model.score(X) == pytest.approx(hmm.training_score)
    smoothed_last = hmm.model.predict_proba(X)[-1]
    np.testing.assert_allclose(hmm.filtered_posteriors(df).iloc[-1], smoothed_last, atol=1e-8)


def test_same_optimum_from_different_seeds_gives_identical_states(switching_ohlcv):
    """
    Canonical state ordering: restarts with different seeds that reach the same
    optimum (here best restart seed 5 vs 50) yield identical state ints and labels,
    not a permutation of them.
    """
    df = switching_ohlcv(n=600)
    a = TrueHMMDetector(n_states=3, random_state=0).fit(df)
    b = TrueHMMDetector(n_states=3, random_state=50).fit(df)
    assert a.get_training_convergence()["best_seed"] != b.get_training_convergence()["best_seed"]
    assert a.training_score == pytest.approx(b.training_score, abs=1e-3)
    np.testing.assert_array_equal(a.predict_with_states(df)[1], b.predict_with_states(df)[1])
    assert a.get_state_regime_map() == b.get_state_regime_map()


def test_labels_stable_across_refits(switching_ohlcv):
    df = switching_ohlcv(n=600)
    first = TrueHMMDetector(n_states=3).fit(df)
    second = TrueHMMDetector(n_states=3).fit(df.copy())
    assert first.get_state_regime_map() == second.get_state_regime_map()
    assert _labels(first, df) == _labels(second, df)


def test_volatility_regimes_are_recovered(switching_ohlcv):
    """The volatile bear regime maps to HIGH_VOLATILITY, calm states do not."""
    df = switching_ohlcv(n=600)
    det = TrueHMMDetector(n_states=3).fit(df)
    regimes = det.get_state_regime_map()
    assert regimes[2] == MarketRegime.HIGH_VOLATILITY
    assert MarketRegime.HIGH_VOLATILITY not in (regimes[0], regimes[1])


def test_homogeneous_market_not_forced_into_vol_buckets():
    """A constant-volatility random walk no longer gets 2 HIGH_VOL + 2 LOW_VOL states."""
    from mra_lib.data_providers import MarketDataProvider

    df = MarketDataProvider.create_provider("mock").fetch("SPY", "2y", "1d")
    regimes = TrueHMMDetector().fit(df).get_state_regime_map().values()
    vol_labels = [
        r for r in regimes if r in {MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY}
    ]
    assert len(vol_labels) <= 1


def test_filtered_posteriors_are_causal(switching_ohlcv):
    df = switching_ohlcv(n=500)
    det = TrueHMMDetector(n_states=3, n_init=3).fit(df.iloc[:300])
    full = det.filtered_posteriors(df)
    prefix = det.filtered_posteriors(df.iloc[:400])
    pd.testing.assert_frame_equal(full.loc[prefix.index], prefix)
    np.testing.assert_allclose(full.sum(axis=1), 1.0)


def test_predict_with_states_matches_predict_regime(fitted):
    hmm, df = fitted
    regime, states, conf = hmm.predict_with_states(df)
    assert (regime, int(states[-1]), conf) == hmm.predict_regime(df)
    assert len(states) == len(hmm._prepare_features(df))


def test_viterbi_option_still_supported(fitted):
    hmm, df = fitted
    regime, state, conf = hmm.predict_regime(df, use_viterbi=True)
    assert regime == hmm.get_state_regime_map()[state]
    assert 0.0 <= conf <= 1.0


def test_confidence_not_saturated(switching_ohlcv):
    """With the stationary feature set, confidence is no longer ~1.0 on every bar."""
    df = switching_ohlcv(n=600)
    conf = TrueHMMDetector().fit(df).filtered_posteriors(df).max(axis=1)
    assert np.percentile(conf, 5) < 0.99


def test_convergence_info_uses_own_check(fitted):
    hmm, _ = fitted
    info = hmm.get_training_convergence()
    assert {"converged", "n_iterations", "best_seed", "max_loglik_decrease"} <= set(info)
    assert info["n_parameters"] == hmm.n_parameters


def test_non_convergence_is_logged(synthetic_ohlcv, caplog):
    df = synthetic_ohlcv(n=300)
    with caplog.at_level(logging.WARNING, logger="mra_lib.indicators.true_hmm_detector"):
        TrueHMMDetector(n_states=4, n_iter=2, n_init=1, tol=1e-12).fit(df)
    assert any("did not converge" in r.getMessage() for r in caplog.records)


def test_select_n_states_by_bic(switching_ohlcv):
    df = switching_ohlcv(n=600)
    best, scores = select_n_states(df, candidates=[1, 2, 3], n_init=2)
    assert set(scores) == {1, 2, 3}
    assert best == min(scores, key=lambda n: scores[n])
    assert best >= 2  # the data has distinct volatility regimes


def test_select_n_states_skips_unsupported_sizes(synthetic_ohlcv):
    df = synthetic_ohlcv(n=150)
    _, scores = select_n_states(df, candidates=[2, 12], n_init=1)
    assert set(scores) == {2}
    with pytest.raises(ValueError, match="No candidate"):
        select_n_states(df, candidates=[12], n_init=1)


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError, match="covariance_type"):
        TrueHMMDetector(covariance_type="bogus")
    with pytest.raises(ValueError, match="n_init"):
        TrueHMMDetector(n_init=0)
