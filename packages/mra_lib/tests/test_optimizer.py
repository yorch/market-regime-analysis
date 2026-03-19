"""Tests for backtesting/optimizer.py — StrategyOptimizer and OptimizationResult."""

from mra_lib.backtesting.optimizer import OptimizationResult, StrategyOptimizer


class TestOptimizationResultScore:
    def _make_result(self, **overrides):
        defaults = {
            "params": {},
            "sharpe": 1.0,
            "total_return": 0.10,
            "excess_return": 0.05,
            "trade_win_rate": 0.55,
            "profit_factor": 1.5,
            "total_trades": 50,
            "max_drawdown": -0.10,
            "window_win_rate": 0.6,
        }
        defaults.update(overrides)
        return OptimizationResult(**defaults)

    def test_score_positive_for_good_results(self):
        r = self._make_result()
        assert r.score > 0

    def test_higher_sharpe_higher_score(self):
        r1 = self._make_result(sharpe=0.5)
        r2 = self._make_result(sharpe=2.0)
        assert r2.score > r1.score

    def test_excess_return_boosts_score(self):
        r1 = self._make_result(excess_return=0.0)
        r2 = self._make_result(excess_return=0.10)
        assert r2.score > r1.score

    def test_deep_drawdown_penalized(self):
        r1 = self._make_result(max_drawdown=-0.10)
        r2 = self._make_result(max_drawdown=-0.40)
        assert r1.score > r2.score

    def test_drawdown_penalty_threshold(self):
        # DD at exactly -20% should have no penalty
        r_ok = self._make_result(max_drawdown=-0.20)
        # DD at -30% should be penalized
        r_bad = self._make_result(max_drawdown=-0.30)
        assert r_ok.score > r_bad.score

    def test_few_trades_penalized(self):
        r1 = self._make_result(total_trades=50)
        r2 = self._make_result(total_trades=10)
        assert r1.score > r2.score

    def test_trade_penalty_threshold_at_15(self):
        r_ok = self._make_result(total_trades=15)
        r_bad = self._make_result(total_trades=14)
        assert r_ok.score > r_bad.score

    def test_negative_sharpe_negative_score(self):
        r = self._make_result(sharpe=-1.0, excess_return=-0.05)
        assert r.score < 0

    def test_results_sortable(self):
        results = [
            self._make_result(sharpe=0.5),
            self._make_result(sharpe=2.0),
            self._make_result(sharpe=1.0),
        ]
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        assert sorted_results[0].sharpe == 2.0
        assert sorted_results[-1].sharpe == 0.5


class TestStrategyOptimizerInit:
    def test_default_init(self):
        import numpy as np
        import pandas as pd

        idx = pd.bdate_range("2020-01-01", periods=500)
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "Open": 100 + np.cumsum(rng.normal(0, 0.5, 500)),
                "High": 101 + np.cumsum(rng.normal(0, 0.5, 500)),
                "Low": 99 + np.cumsum(rng.normal(0, 0.5, 500)),
                "Close": 100 + np.cumsum(rng.normal(0, 0.5, 500)),
                "Volume": rng.integers(1_000_000, 10_000_000, 500),
            },
            index=idx,
        )
        opt = StrategyOptimizer(df, min_train_days=252, test_days=63)
        assert opt.min_train_days == 252
        assert opt.test_days == 63
        assert opt.results == []

    def test_custom_params(self):
        import pandas as pd

        df = pd.DataFrame({"Close": [1, 2, 3]})
        opt = StrategyOptimizer(
            df,
            min_train_days=100,
            test_days=30,
            anchored=False,
            initial_capital=50_000,
            n_hmm_states=3,
            hmm_n_iter=20,
            retrain_frequency=10,
        )
        assert opt.anchored is False
        assert opt.initial_capital == 50_000
        assert opt.n_hmm_states == 3


class TestDefaultGridAndRanges:
    def test_default_grid(self):
        import pandas as pd

        df = pd.DataFrame({"Close": [1]})
        opt = StrategyOptimizer(df)
        grid = opt._default_grid()
        assert "bull_mult" in grid
        assert "bear_mult" in grid
        assert "stop_loss" in grid
        assert isinstance(grid["bull_mult"], list)

    def test_default_ranges(self):
        import pandas as pd

        df = pd.DataFrame({"Close": [1]})
        opt = StrategyOptimizer(df)
        ranges = opt._default_ranges()
        assert "bull_mult" in ranges
        assert isinstance(ranges["bull_mult"], tuple)
        assert len(ranges["bull_mult"]) == 2


class TestFormatParams:
    def test_format_float(self):
        result = StrategyOptimizer._format_params({"a": 1.234, "b": 5})
        assert "a=1.23" in result
        assert "b=5" in result

    def test_format_empty(self):
        result = StrategyOptimizer._format_params({})
        assert result == ""


class TestPrintTopResults:
    def test_no_results(self, capsys):
        import pandas as pd

        df = pd.DataFrame({"Close": [1]})
        opt = StrategyOptimizer(df)
        opt.print_top_results()
        captured = capsys.readouterr()
        assert "No results to display" in captured.out

    def test_with_results(self, capsys):
        import pandas as pd

        df = pd.DataFrame({"Close": [1]})
        opt = StrategyOptimizer(df)
        opt.results = [
            OptimizationResult(
                params={
                    "stop_loss": 0.05,
                    "bull_mult": 1.5,
                    "bear_mult": 0.5,
                    "base_fraction": 0.1,
                },
                sharpe=1.2,
                total_return=0.15,
                excess_return=0.05,
                trade_win_rate=0.55,
                profit_factor=1.8,
                total_trades=40,
                max_drawdown=-0.12,
                window_win_rate=0.6,
            )
        ]
        opt.print_top_results(n=5)
        captured = capsys.readouterr()
        assert "TOP OPTIMIZATION RESULTS" in captured.out
        assert "BEST PARAMETERS" in captured.out

    def test_missing_param_keys(self, capsys):
        import pandas as pd

        df = pd.DataFrame({"Close": [1]})
        opt = StrategyOptimizer(df)
        opt.results = [
            OptimizationResult(
                params={"custom_param": 42},
                sharpe=0.5,
                total_return=0.05,
                excess_return=0.01,
                trade_win_rate=0.5,
                profit_factor=1.0,
                total_trades=20,
                max_drawdown=-0.05,
                window_win_rate=0.5,
            )
        ]
        # Should handle missing stop_loss/bull_mult gracefully
        opt.print_top_results()
        captured = capsys.readouterr()
        assert "?" in captured.out
