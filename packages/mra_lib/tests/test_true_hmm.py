"""TrueHMMDetector (hmmlearn) vs GMM detector on deterministic synthetic data."""

import numpy as np
import pytest

from mra_lib.config.enums import MarketRegime
from mra_lib.indicators.hmm_detector import HiddenMarkovRegimeDetector
from mra_lib.indicators.true_hmm_detector import TrueHMMDetector


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


def test_true_hmm_compare_with_gmm_reports_both_predictions(fitted):
    hmm, df = fitted
    gmm = HiddenMarkovRegimeDetector(n_states=6).fit(df)

    comparison = hmm.compare_with_gmm(df, gmm)

    valid = {r.value for r in MarketRegime}
    assert comparison["hmm_regime"] in valid
    assert comparison["gmm_regime"] in valid
    assert comparison["regime_agreement"] == (comparison["hmm_regime"] == comparison["gmm_regime"])


def test_true_hmm_predict_before_fit_raises(synthetic_ohlcv):
    with pytest.raises(ValueError, match="fitted"):
        TrueHMMDetector(n_states=4).predict_regime(synthetic_ohlcv(n=100))
