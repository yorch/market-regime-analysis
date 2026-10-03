"""Tests for walk-forward regime caching/stitching and optimizer evaluation hygiene."""

import logging
import math

import numpy as np
import pandas as pd
import pytest

from mra_lib.backtesting import walk_forward as wf_module
from mra_lib.backtesting.calibrator import RegimeMultiplierCalibrator
from mra_lib.backtesting.metrics import PerformanceMetrics
from mra_lib.backtesting.optimizer import StrategyOptimizer
from mra_lib.backtesting.strategy import RegimeStrategy
from mra_lib.backtesting.trade_stats import compute_pnl_stats, finite_profit_factor
from mra_lib.backtesting.walk_forward import RegimeCache, WalkForwardValidator


def _synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.005,
            "Low": np.minimum(open_, close) * 0.995,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=pd.bdate_range("2018-01-01", periods=n),
    )


FAST_HMM = {"n_hmm_states": 3, "hmm_n_iter": 10, "min_train_days": 252, "test_days": 63}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return _synthetic()


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


class TestRegimeDetection:
    def test_fast_filter_matches_per_bar_prediction(self, data, monkeypatch):
        v = WalkForwardValidator(RegimeStrategy(), retrain_frequency=20, **FAST_HMM)
        fast_r, fast_c = v._detect_regimes_walk_forward(data, 252, 315)

        def _disabled(*args, **kwargs):
            raise RuntimeError("fast path disabled")

        monkeypatch.setattr(WalkForwardValidator, "_filtered_segment", staticmethod(_disabled))
        slow_r, slow_c = v._detect_regimes_walk_forward(data, 252, 315)

        assert list(fast_r) == list(slow_r)
        np.testing.assert_allclose(fast_c.to_numpy(), slow_c.to_numpy(), atol=1e-6)

    def test_failed_refit_keeps_previous_model(self, data, monkeypatch, caplog):
        v = WalkForwardValidator(RegimeStrategy(), retrain_frequency=20, **FAST_HMM)
        real_fit = WalkForwardValidator._fit_detector
        calls = {"n": 0}

        def flaky_fit(self, train_df):
            calls["n"] += 1
            if calls["n"] > 1:
                raise ValueError("boom")
            return real_fit(self, train_df)

        monkeypatch.setattr(WalkForwardValidator, "_fit_detector", flaky_fit)
        with caplog.at_level(logging.WARNING, logger=wf_module.__name__):
            _, confs = v._detect_regimes_walk_forward(data, 252, 315)

        assert calls["n"] == 4  # refits at 252, 272, 292, 312
        # Bars after the failed refits are still predicted by the first model
        assert (confs.iloc[20:] > 0).all()
        assert "keeping previous model" in caplog.text

    def test_regime_cache_reused_across_strategies(self, data, monkeypatch):
        v = WalkForwardValidator(RegimeStrategy(), retrain_frequency=63, **FAST_HMM)
        cache = v.compute_regimes(data)
        assert cache.matches(data)
        assert set(cache.windows) == set(v.window_bounds(len(data)))

        def _no_detection(*args, **kwargs):
            raise AssertionError("regimes should come from the cache")

        monkeypatch.setattr(WalkForwardValidator, "_detect_regimes_walk_forward", _no_detection)
        for mult in (0.5, 1.5):
            other = WalkForwardValidator(
                RegimeStrategy.from_param_vector({"bull_mult": mult}),
                retrain_frequency=63,
                **FAST_HMM,
            )
            res = other.run(data, verbose=False, regime_cache=cache)
            assert res["n_windows"] == 3

    def test_mismatched_cache_ignored(self, data):
        cache = RegimeCache(n_rows=10, first_index=None, last_index=None)
        assert not cache.matches(data)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _window(values, initial=100.0):
    eq = pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))
    perf = PerformanceMetrics([], eq, initial_capital=initial)
    return {
        "strategy_return": values[-1] / initial - 1,
        "buy_hold_return": 0.0,
        "trades": 0,
        "test_days": len(values),
        "performance": perf,
    }


class TestAggregation:
    def test_stitched_max_drawdown_spans_windows(self):
        v = WalkForwardValidator(RegimeStrategy(), initial_capital=100.0)
        windows = [_window([110.0, 120.0, 108.0]), _window([95.0, 90.0])]
        res = v._aggregate_results(windows, pd.DataFrame())
        # 120 peak -> 108 * 0.90 = 97.2 => -19%, deeper than any single window (-10%)
        assert res["max_drawdown"] == pytest.approx(97.2 / 120 - 1)
        assert res["worst_window_drawdown"] == pytest.approx(-0.10)
        assert res["stitched_equity_curve"].iloc[-1] == pytest.approx(97.2)
        assert res["compounded_strategy_return"] == pytest.approx(97.2 / 100 - 1)

    def test_partial_window_weighted_by_length(self):
        v = WalkForwardValidator(RegimeStrategy(), initial_capital=100.0)
        res = v._aggregate_results(
            [_window(list(np.linspace(101, 110, 63))), _window([101.0, 102.0])],
            pd.DataFrame(),
        )
        assert res["total_test_days"] == 65
        assert res["years"] == pytest.approx(65 / 252)


# ---------------------------------------------------------------------------
# Trade stats
# ---------------------------------------------------------------------------


class TestTradeStats:
    def test_profit_factor_sentinels(self):
        assert compute_pnl_stats([1.0, 2.0])["profit_factor"] == math.inf
        assert math.isnan(compute_pnl_stats([])["profit_factor"])
        assert compute_pnl_stats([3.0, -1.0])["profit_factor"] == pytest.approx(3.0)

    def test_finite_profit_factor(self):
        assert finite_profit_factor(math.inf) == 100.0
        assert finite_profit_factor(math.nan) == 0.0
        assert finite_profit_factor(2.5) == 2.5


# ---------------------------------------------------------------------------
# Strategy params
# ---------------------------------------------------------------------------


class TestStrategyParams:
    def test_unknown_param_key_raises(self):
        with pytest.raises(ValueError, match="stoploss"):
            RegimeStrategy.from_param_vector({"stoploss": 0.05})

    def test_canonical_params_merges_irrelevant_bear_settings(self):
        a = RegimeStrategy.canonical_params({"bear_mult": 0.0, "bear_short": 1})
        b = RegimeStrategy.canonical_params({"bear_mult": 0.5, "bear_short": 0})
        c = RegimeStrategy.canonical_params({"bear_mult": 0.5, "bear_short": 1})
        assert a == b
        assert c != a


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class TestOptimizer:
    def test_hmm_fit_once_per_window_across_param_sets(self, data, monkeypatch):
        fits = {"n": 0}
        real_fit = wf_module.TrueHMMDetector.fit

        def counting_fit(self, df):
            fits["n"] += 1
            return real_fit(self, df)

        monkeypatch.setattr(wf_module.TrueHMMDetector, "fit", counting_fit)
        opt = StrategyOptimizer(data, retrain_frequency=63, **FAST_HMM)
        results = opt.grid_search(
            {"bull_mult": [1.0, 1.5, 2.0], "stop_loss": [0.03, 0.05]}, verbose=False
        )
        n_windows = len(WalkForwardValidator(RegimeStrategy(), **FAST_HMM).window_bounds(400))
        assert len(results) == 6
        assert opt.n_trials == 6
        assert fits["n"] == n_windows == 3

    def test_grid_skips_equivalent_combinations(self, monkeypatch):
        opt = StrategyOptimizer(_synthetic(400), **FAST_HMM)
        monkeypatch.setattr(
            WalkForwardValidator,
            "compute_regimes",
            lambda self, df, start_index=None: RegimeCache(len(df), df.index[0], df.index[-1]),
        )
        seen = []
        monkeypatch.setattr(opt, "_evaluate_params", lambda p, verbose=False: seen.append(p))
        opt.grid_search({"bear_mult": [0.0, 0.5, 1.0], "bear_short": [0, 1]}, verbose=False)
        assert len(seen) == 3
        assert opt.n_skipped_duplicates == 3

    def test_unknown_grid_key_raises_before_search(self):
        opt = StrategyOptimizer(_synthetic(400), **FAST_HMM)
        with pytest.raises(ValueError, match="Unknown strategy parameter"):
            opt.grid_search({"bull_mutl": [1.0]}, verbose=False)

    def test_random_search_seed_reproducible(self, monkeypatch):
        monkeypatch.setattr(
            WalkForwardValidator,
            "compute_regimes",
            lambda self, df, start_index=None: RegimeCache(len(df), df.index[0], df.index[-1]),
        )

        def sample(seed):
            opt = StrategyOptimizer(_synthetic(400), **FAST_HMM)
            seen = []
            monkeypatch.setattr(opt, "_evaluate_params", lambda p, verbose=False: seen.append(p))
            opt.random_search(n_iterations=5, verbose=False, seed=seed)
            return seen

        assert sample(7) == sample(7)
        assert sample(7) != sample(8)

    def test_failures_are_counted_and_logged(self, monkeypatch, caplog):
        opt = StrategyOptimizer(_synthetic(400), **FAST_HMM)
        monkeypatch.setattr(
            WalkForwardValidator,
            "compute_regimes",
            lambda self, df, start_index=None: RegimeCache(len(df), df.index[0], df.index[-1]),
        )

        def boom(self, *args, **kwargs):
            raise RuntimeError("kaput")

        monkeypatch.setattr(WalkForwardValidator, "run", boom)
        with caplog.at_level(logging.WARNING):
            results = opt.grid_search({"bull_mult": [1.0, 2.0]}, verbose=False)
        assert results == []
        assert opt.n_failures == 2
        assert opt.n_trials == 2
        assert "kaput" in caplog.text

    def test_holdout_split_and_evaluation(self, monkeypatch):
        df = _synthetic(500)
        opt = StrategyOptimizer(df, holdout_frac=0.2, retrain_frequency=63, **FAST_HMM)
        assert opt.split_index == 400
        assert len(opt.search_df) == 400
        assert opt.has_holdout
        summary = opt.search_summary()
        assert summary["holdout_bars"] == 100

        res = opt.evaluate_holdout({"bull_mult": 1.0})
        assert res is not None
        first_test = res["window_results"][0]["test_start"]
        assert first_test == df.index[400]
        assert res["total_test_days"] == 100

    def test_no_holdout_returns_none(self):
        opt = StrategyOptimizer(_synthetic(400), **FAST_HMM)
        assert not opt.has_holdout
        assert opt.evaluate_holdout({}) is None

    def test_invalid_holdout_frac(self):
        with pytest.raises(ValueError):
            StrategyOptimizer(_synthetic(400), holdout_frac=1.0)

    def test_search_period_too_short_raises(self):
        opt = StrategyOptimizer(_synthetic(400), holdout_frac=0.5, **FAST_HMM)
        with pytest.raises(ValueError, match="Search period"):
            opt.grid_search({"bull_mult": [1.0]}, verbose=False)


class TestCalibratorHoldout:
    def test_in_sample_flag_and_holdout_metrics(self):
        df = _synthetic(500)
        settings = dict(FAST_HMM)
        cal = RegimeMultiplierCalibrator(
            df, holdout_frac=0.2, retrain_frequency=63, min_trades_per_regime=1, **settings
        )
        result = cal.calibrate_with_details(verbose=False)
        assert result.total_trades > 0
        assert result.in_sample is False
        assert result.holdout_metrics is not None
        assert result.holdout_metrics["total_test_days"] == 100

        cal_is = RegimeMultiplierCalibrator(
            df.iloc[:400], retrain_frequency=63, min_trades_per_regime=1, **settings
        )
        assert cal_is.calibrate_with_details(verbose=False).in_sample is True
