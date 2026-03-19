"""Tests for RegimeMultiplierCalibrator."""

import numpy as np
import pandas as pd
import pytest

from mra_lib.backtesting.calibrator import (
    CalibrationResult,
    RegimeMultiplierCalibrator,
    RegimeTradeStats,
)
from mra_lib.backtesting.strategy import RegimeStrategy
from mra_lib.config.enums import MarketRegime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_ohlcv(n: int = 450, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with enough bars for walk-forward."""
    rng = np.random.RandomState(seed)
    # Create regime-like patterns: trending + volatile + mean-reverting
    seg = n // 3
    trend = np.cumsum(rng.normal(0.002, 0.008, seg))
    volatile = np.cumsum(rng.normal(0.0, 0.025, seg))
    mean_rev = np.cumsum(rng.normal(0.0, 0.005, n - 2 * seg))
    log_prices = np.concatenate([trend, volatile, mean_rev])
    close = 100 * np.exp(log_prices)

    dates = pd.bdate_range("2019-01-01", periods=n)
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


# ---------------------------------------------------------------------------
# Tests: RegimeTradeStats dataclass
# ---------------------------------------------------------------------------


class TestRegimeTradeStats:
    """Tests for the per-regime stats dataclass."""

    def test_defaults(self):
        rs = RegimeTradeStats(regime=MarketRegime.BULL_TRENDING)
        assert rs.n_trades == 0
        assert rs.total_pnl == 0.0
        assert rs.sharpe == 0.0

    def test_fields_stored(self):
        rs = RegimeTradeStats(
            regime=MarketRegime.BEAR_TRENDING,
            n_trades=10,
            win_rate=0.6,
            sharpe=1.5,
        )
        assert rs.regime == MarketRegime.BEAR_TRENDING
        assert rs.n_trades == 10
        assert rs.win_rate == 0.6
        assert rs.sharpe == 1.5


# ---------------------------------------------------------------------------
# Tests: CalibrationResult dataclass
# ---------------------------------------------------------------------------


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_creation(self):
        mults = dict.fromkeys(MarketRegime, 1.0)
        stats = {r: RegimeTradeStats(regime=r) for r in MarketRegime}
        result = CalibrationResult(
            multipliers=mults,
            regime_stats=stats,
            method="sharpe_weighted",
            total_trades=50,
            trades_per_regime=dict.fromkeys(MarketRegime, 0),
            baseline_sharpe=0.5,
        )
        assert result.method == "sharpe_weighted"
        assert result.total_trades == 50


# ---------------------------------------------------------------------------
# Tests: _compute_regime_stats (unit test on the stats computation)
# ---------------------------------------------------------------------------


class TestComputeRegimeStats:
    """Test per-regime stats computation in isolation."""

    def test_groups_trades_by_regime(self):
        df = _make_synthetic_ohlcv(450)
        cal = RegimeMultiplierCalibrator(df=df)

        trades = [
            {"entry_regime": "Bull Trending", "pnl": 100, "holding_days": 5},
            {"entry_regime": "Bull Trending", "pnl": -50, "holding_days": 3},
            {"entry_regime": "Bear Trending", "pnl": 200, "holding_days": 7},
        ]

        stats = cal._compute_regime_stats(trades)

        assert stats[MarketRegime.BULL_TRENDING].n_trades == 2
        assert stats[MarketRegime.BEAR_TRENDING].n_trades == 1
        assert stats[MarketRegime.MEAN_REVERTING].n_trades == 0

    def test_win_rate_calculation(self):
        df = _make_synthetic_ohlcv(450)
        cal = RegimeMultiplierCalibrator(df=df)

        trades = [
            {"entry_regime": "Bull Trending", "pnl": 100, "holding_days": 5},
            {"entry_regime": "Bull Trending", "pnl": 50, "holding_days": 3},
            {"entry_regime": "Bull Trending", "pnl": -30, "holding_days": 4},
        ]

        stats = cal._compute_regime_stats(trades)
        assert stats[MarketRegime.BULL_TRENDING].win_rate == pytest.approx(2 / 3)

    def test_profit_factor_calculation(self):
        df = _make_synthetic_ohlcv(450)
        cal = RegimeMultiplierCalibrator(df=df)

        trades = [
            {"entry_regime": "Bull Trending", "pnl": 200, "holding_days": 5},
            {"entry_regime": "Bull Trending", "pnl": -100, "holding_days": 3},
        ]

        stats = cal._compute_regime_stats(trades)
        assert stats[MarketRegime.BULL_TRENDING].profit_factor == pytest.approx(2.0)

    def test_unknown_regime_string_maps_correctly(self):
        df = _make_synthetic_ohlcv(450)
        cal = RegimeMultiplierCalibrator(df=df)

        trades = [
            {"entry_regime": "Unknown", "pnl": 50, "holding_days": 2},
            {"entry_regime": "INVALID_REGIME", "pnl": 30, "holding_days": 1},
        ]

        stats = cal._compute_regime_stats(trades)
        assert stats[MarketRegime.UNKNOWN].n_trades == 2  # both map to UNKNOWN


# ---------------------------------------------------------------------------
# Tests: _score_regimes and _normalize_to_multipliers
# ---------------------------------------------------------------------------


class TestScoringAndNormalization:
    """Test the scoring and normalization pipeline."""

    def test_unknown_always_zero(self):
        df = _make_synthetic_ohlcv(450)
        cal = RegimeMultiplierCalibrator(df=df, min_trades_per_regime=1)

        stats = {
            MarketRegime.UNKNOWN: RegimeTradeStats(
                regime=MarketRegime.UNKNOWN,
                n_trades=10,
                sharpe=5.0,
            ),
        }
        scores = cal._score_regimes(stats, "sharpe_weighted")
        assert scores[MarketRegime.UNKNOWN] == 0.0

    def test_too_few_trades_get_zero_score(self):
        df = _make_synthetic_ohlcv(450)
        cal = RegimeMultiplierCalibrator(df=df, min_trades_per_regime=10)

        stats = {
            MarketRegime.BULL_TRENDING: RegimeTradeStats(
                regime=MarketRegime.BULL_TRENDING,
                n_trades=3,
                sharpe=2.0,
            ),
        }
        scores = cal._score_regimes(stats, "sharpe_weighted")
        assert scores[MarketRegime.BULL_TRENDING] == 0.0

    def test_normalization_max_is_two(self):
        df = _make_synthetic_ohlcv(450)
        cal = RegimeMultiplierCalibrator(df=df)

        scores = {
            MarketRegime.BULL_TRENDING: 3.0,
            MarketRegime.BEAR_TRENDING: 1.0,
            MarketRegime.UNKNOWN: 0.0,
        }
        mults = cal._normalize_to_multipliers(scores)
        assert mults[MarketRegime.BULL_TRENDING] == pytest.approx(2.0)
        assert mults[MarketRegime.BEAR_TRENDING] == pytest.approx(0.5)
        assert mults[MarketRegime.UNKNOWN] == 0.0

    def test_normalization_equal_scores(self):
        df = _make_synthetic_ohlcv(450)
        cal = RegimeMultiplierCalibrator(df=df)

        scores = {
            MarketRegime.BULL_TRENDING: 1.0,
            MarketRegime.MEAN_REVERTING: 1.0,
        }
        mults = cal._normalize_to_multipliers(scores)
        assert mults[MarketRegime.BULL_TRENDING] == pytest.approx(1.0)
        assert mults[MarketRegime.MEAN_REVERTING] == pytest.approx(1.0)

    def test_normalization_all_zero(self):
        df = _make_synthetic_ohlcv(450)
        cal = RegimeMultiplierCalibrator(df=df)

        scores = dict.fromkeys(MarketRegime, 0.0)
        mults = cal._normalize_to_multipliers(scores)
        for r in MarketRegime:
            assert mults[r] == 0.0

    def test_all_methods_accepted(self):
        df = _make_synthetic_ohlcv(450)
        cal = RegimeMultiplierCalibrator(df=df, min_trades_per_regime=1)

        stats = {
            MarketRegime.BULL_TRENDING: RegimeTradeStats(
                regime=MarketRegime.BULL_TRENDING,
                n_trades=10,
                sharpe=1.5,
                win_rate=0.7,
                profit_factor=2.0,
                kelly_fraction=0.15,
            ),
        }

        for method in ("sharpe_weighted", "win_rate", "profit_factor", "kelly"):
            scores = cal._score_regimes(stats, method)
            assert MarketRegime.BULL_TRENDING in scores

    def test_invalid_method_raises(self):
        df = _make_synthetic_ohlcv(450)
        cal = RegimeMultiplierCalibrator(df=df)
        with pytest.raises(ValueError, match="Unknown"):
            cal._score_regimes({}, "invalid_method")


# ---------------------------------------------------------------------------
# Tests: Full calibration (integration, uses walk-forward)
# ---------------------------------------------------------------------------


class TestCalibrateIntegration:
    """Integration tests that run the full calibration pipeline."""

    @pytest.fixture
    def calibrator(self):
        """Calibrator with small data for fast tests."""
        df = _make_synthetic_ohlcv(450)
        return RegimeMultiplierCalibrator(
            df=df,
            n_hmm_states=3,
            hmm_n_iter=20,
            retrain_frequency=30,
            min_train_days=200,
            test_days=50,
            min_trades_per_regime=2,
        )

    def test_calibrate_returns_dict(self, calibrator):
        mults = calibrator.calibrate(method="sharpe_weighted", verbose=False)
        assert isinstance(mults, dict)

    def test_calibrate_multipliers_non_negative(self, calibrator):
        mults = calibrator.calibrate(method="sharpe_weighted", verbose=False)
        for r, m in mults.items():
            assert m >= 0.0, f"{r.value}: {m}"

    def test_calibrate_unknown_is_zero(self, calibrator):
        mults = calibrator.calibrate(method="sharpe_weighted", verbose=False)
        assert mults.get(MarketRegime.UNKNOWN, 0.0) == 0.0

    def test_calibrate_with_details_returns_result(self, calibrator):
        result = calibrator.calibrate_with_details(
            method="sharpe_weighted",
            verbose=False,
        )
        assert isinstance(result, CalibrationResult)
        assert result.method == "sharpe_weighted"
        assert result.total_trades >= 0

    def test_create_calibrated_strategy(self, calibrator):
        strategy = calibrator.create_calibrated_strategy(
            method="sharpe_weighted",
            verbose=False,
        )
        assert isinstance(strategy, RegimeStrategy)
        # The strategy should have calibrated multipliers
        for r in MarketRegime:
            assert r in strategy.regime_multipliers

    def test_invalid_method_raises(self, calibrator):
        with pytest.raises(ValueError, match="Unknown method"):
            calibrator.calibrate(method="not_a_method")
