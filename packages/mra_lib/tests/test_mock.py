"""Core functionality smoke tests on deterministic synthetic data (no network)."""

import pandas as pd

from mra_lib.config.enums import MarketRegime
from mra_lib.indicators.hmm_detector import HiddenMarkovRegimeDetector
from mra_lib.risk.risk_calculator import SimonsRiskCalculator


def test_gmm_detector_fits_and_predicts_on_synthetic_data(synthetic_ohlcv):
    df = synthetic_ohlcv(n=365)
    hmm = HiddenMarkovRegimeDetector(n_states=6)

    hmm.fit(df)
    regime, state, confidence = hmm.predict_regime(df)

    assert isinstance(regime, MarketRegime)
    assert 0 <= state < 6
    assert 0.0 <= confidence <= 1.0


def test_gmm_detector_feature_engineering_produces_finite_features(synthetic_ohlcv):
    df = synthetic_ohlcv(n=365)
    features = HiddenMarkovRegimeDetector(n_states=6)._prepare_features(df)

    assert isinstance(features, pd.DataFrame)
    assert len(features.columns) > 5
    assert len(features) > 0
    assert not features.isna().any().any()


def test_risk_sizes_are_bounded_for_detected_regime(synthetic_ohlcv):
    df = synthetic_ohlcv(n=365)
    hmm = HiddenMarkovRegimeDetector(n_states=6).fit(df)
    regime, _, confidence = hmm.predict_regime(df)

    kelly = SimonsRiskCalculator.calculate_kelly_optimal_size(
        win_rate=0.55, avg_win=0.02, avg_loss=0.015, confidence=0.8
    )
    regime_size = SimonsRiskCalculator.calculate_regime_adjusted_size(
        base_size=0.02, regime=regime, confidence=confidence, persistence=0.7
    )

    assert 0.0 <= kelly <= 1.0
    assert 0.0 <= regime_size <= 1.0
