"""Tests for the one-call backtest helper (mra_lib.backtesting.runner)."""

import json
import math

import pandas as pd
import pytest

from mra_lib.backtesting import (
    COST_MODELS,
    RegimeStrategy,
    StrategyParams,
    TransactionCostModel,
    WalkForwardValidator,
    load_strategy_params,
    make_cost_model,
    parse_strategy_params,
    run_backtest,
    validate_strategy_params,
)
from mra_lib.backtesting.runner import (
    _FLAG_PARAMS,
    _MULTIPLIER_PARAMS,
    _OPTIONAL_PARAMS,
    TRADE_COLUMNS,
    _buy_hold,
    _exposure,
)
from mra_lib.backtesting.strategy import PARAM_KEYS
from mra_lib.errors import InsufficientDataError, InvalidParametersError, MRAError

# Small, fast walk-forward settings shared by the tests
FAST = {"train_bars": 120, "test_bars": 40, "n_hmm_states": 2, "hmm_n_iter": 20}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    from .conftest import make_switching_ohlcv

    return make_switching_ohlcv(n=260, seed=3)


@pytest.fixture(scope="module")
def wf_report(df):
    return run_backtest(df, mode="walk-forward", symbol="TEST", **FAST)


@pytest.fixture(scope="module")
def simple_report(df):
    return run_backtest(df, mode="simple", symbol="TEST", n_hmm_states=2, hmm_n_iter=20)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestParams:
    def test_params_every_key_has_a_rule(self):
        ranged = {"base_fraction", "max_position", "min_confidence"}
        assert _MULTIPLIER_PARAMS | _FLAG_PARAMS | _OPTIONAL_PARAMS | ranged == PARAM_KEYS

    def test_params_valid_roundtrip(self):
        params = {"bull_mult": 1.5, "bear_short": 0, "stop_loss": None, "confidence_scaling": True}
        assert validate_strategy_params(params) == params

    def test_params_unknown_key_listed(self):
        with pytest.raises(InvalidParametersError, match="stoploss"):
            validate_strategy_params({"stoploss": 0.05})

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("bull_mult", "1.5"),
            ("bull_mult", True),
            ("bull_mult", -0.1),
            ("bull_mult", float("nan")),
            ("base_fraction", 0),
            ("max_position", 1.5),
            ("min_confidence", 2),
            ("stop_loss", 1.0),
            ("take_profit", -0.1),
            ("bear_short", 2),
            ("confidence_scaling", "yes"),
            ("bull_mult", None),
        ],
    )
    def test_params_invalid_value(self, key, value):
        with pytest.raises(InvalidParametersError, match=key):
            validate_strategy_params({key: value})

    def test_params_reports_all_problems(self):
        with pytest.raises(InvalidParametersError) as exc:
            validate_strategy_params({"nope": 1, "bull_mult": "x", "max_position": 3})
        msg = str(exc.value)
        assert "nope" in msg and "bull_mult" in msg and "max_position" in msg

    def test_params_error_is_mra_and_value_error(self):
        with pytest.raises(MRAError):
            validate_strategy_params({"nope": 1})
        with pytest.raises(ValueError):
            validate_strategy_params({"nope": 1})

    def test_params_not_an_object(self):
        with pytest.raises(InvalidParametersError, match="JSON object"):
            parse_strategy_params([1, 2])

    def test_params_flat_object(self):
        sp = parse_strategy_params({"bull_mult": 2.0})
        assert sp.params == {"bull_mult": 2.0}
        assert sp.selected_through is None

    def test_params_optimizer_output(self):
        data = {
            "symbol": "SPY",
            "best_params": {"bull_mult": 2.0, "bear_short": 1},
            "search": {"search_end": "2024-06-28 00:00:00"},
            "in_sample": {"sharpe": 1.0},
        }
        sp = parse_strategy_params(data)
        assert sp.params == {"bull_mult": 2.0, "bear_short": 1}
        assert sp.source == "mra-optimize output"
        assert sp.selected_through == pd.Timestamp("2024-06-28")

    def test_params_optimizer_output_invalid_best_params(self):
        with pytest.raises(InvalidParametersError, match="unknown"):
            parse_strategy_params({"best_params": {"wat": 1}})

    def test_params_load_file(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"stop_loss": 0.03}))
        sp = load_strategy_params(path)
        assert sp.params == {"stop_loss": 0.03}
        assert "p.json" in sp.source

    def test_params_load_invalid_json(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text("{not json")
        with pytest.raises(InvalidParametersError, match="not valid JSON"):
            load_strategy_params(path)

    def test_params_build_strategy(self):
        strat = StrategyParams(params={"base_fraction": 0.05, "stop_loss": 0.02}).build_strategy()
        assert isinstance(strat, RegimeStrategy)
        assert strat.base_position_fraction == 0.05
        assert strat.stop_loss_pct == 0.02


class TestCostModels:
    def test_cost_models_presets(self):
        for name in COST_MODELS:
            assert isinstance(make_cost_model(name), TransactionCostModel)

    def test_cost_models_none_is_free(self):
        costs = make_cost_model("none").calculate_total_cost(100.0, 10, "BUY", 1e6)
        assert costs["total_cost"] == 0.0

    def test_cost_models_unknown(self):
        with pytest.raises(InvalidParametersError, match="Unknown cost model"):
            make_cost_model("free")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_buy_hold_chains_segments(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        a = pd.Series([10.0, 11.0], index=idx[:2])
        b = pd.Series([20.0, 18.0], index=idx[2:])
        curve, trades = _buy_hold([a, b], 100.0)
        assert curve.tolist() == pytest.approx([100.0, 110.0, 110.0, 99.0])
        assert [t["pnl"] for t in trades] == pytest.approx([10.0, -11.0])

    def test_exposure_counts_bars_held_at_close(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        closes = pd.Series([10.0] * 5, index=idx)
        equity = pd.Series([100.0] * 5, index=idx)
        trades = [{"entry_date": idx[1], "exit_date": idx[3], "shares": 2}]
        held, exp_sum = _exposure(trades, closes, equity)
        assert held == 2
        assert exp_sum == pytest.approx(0.4)  # 2 bars x 20/100


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------


class TestWalkForward:
    def test_walk_forward_labelled_out_of_sample(self, wf_report):
        assert wf_report.in_sample is False
        assert wf_report.sample_label == "OUT-OF-SAMPLE"
        assert "OUT-OF-SAMPLE" in wf_report.format_report()

    def test_walk_forward_matches_validator(self, df, wf_report):
        validator = WalkForwardValidator(
            strategy=RegimeStrategy.from_param_vector({}),
            n_hmm_states=2,
            hmm_n_iter=20,
            min_train_days=120,
            test_days=40,
        )
        expected = validator.run(df, verbose=False)
        s = wf_report.strategy
        assert s.total_return == pytest.approx(expected["compounded_strategy_return"])
        assert wf_report.buy_hold.total_return == pytest.approx(expected["compounded_bh_return"])
        assert s.sharpe == pytest.approx(expected["sharpe_ratio"])
        assert s.sortino == pytest.approx(expected["sortino_ratio"])
        assert s.max_drawdown == pytest.approx(expected["max_drawdown"])
        assert s.max_drawdown_duration == expected["max_drawdown_duration"]
        assert s.n_trades == expected["total_trades"]
        assert s.win_rate == pytest.approx(expected["trade_win_rate"])
        assert wf_report.n_bars == expected["total_test_days"]
        assert len(wf_report.windows) == expected["n_windows"]

    def test_walk_forward_windows(self, wf_report):
        assert [w.index for w in wf_report.windows] == list(range(1, len(wf_report.windows) + 1))
        assert sum(w.bars for w in wf_report.windows) == wf_report.n_bars
        assert sum(w.trades for w in wf_report.windows) == len(wf_report.trades)
        assert wf_report.n_fits > 0
        assert wf_report.refit_failures == 0
        assert wf_report.start == wf_report.windows[0].test_start

    def test_walk_forward_trades_frame(self, wf_report):
        frame = wf_report.trades_frame()
        assert list(frame.columns[: len(TRADE_COLUMNS) + 1]) == ["window", *TRADE_COLUMNS]
        assert len(frame) == len(wf_report.trades)

    def test_walk_forward_exposure_bounds(self, wf_report):
        s, b = wf_report.strategy, wf_report.buy_hold
        assert 0.0 <= s.time_in_market <= 1.0
        assert 0.0 <= s.avg_exposure <= s.time_in_market
        assert b.time_in_market == 1.0 and b.avg_exposure == 1.0
        assert b.n_trades == len(wf_report.windows)

    def test_walk_forward_to_dict_is_strict_json(self, wf_report):
        data = json.loads(json.dumps(wf_report.to_dict(include_trades=True), allow_nan=False))
        assert data["sample"] == "out-of-sample"
        assert set(data["strategy"]) == set(data["buy_hold"])
        assert len(data["windows"]) == len(wf_report.windows)
        assert len(data["trades"]) == len(wf_report.trades)
        assert data["excess_return"] == pytest.approx(
            data["strategy"]["total_return"] - data["buy_hold"]["total_return"]
        )

    def test_walk_forward_overlap_warning(self, df):
        mid = df.index[200]
        sp = StrategyParams(params={}, source="mra-optimize output", selected_through=mid)
        report = run_backtest(df, sp, "walk-forward", **FAST)
        expected = sum(1 for w in report.windows if w.test_start <= mid)
        assert report.overlapping_windows == expected > 0
        assert any("NOT for the parameters" in w for w in report.warnings)

    def test_walk_forward_params_change_result(self, df, wf_report):
        report = run_backtest(df, {"base_fraction": 0.02, "max_position": 0.05}, **FAST)
        assert report.params == {"base_fraction": 0.02, "max_position": 0.05}
        assert report.strategy.avg_exposure < wf_report.strategy.avg_exposure

    def test_walk_forward_insufficient_data(self, df):
        with pytest.raises(InsufficientDataError, match="at least 160 bars"):
            run_backtest(df.iloc[:150], **FAST)

    def test_walk_forward_train_too_small_for_model(self, df):
        with pytest.raises(InvalidParametersError, match="too small"):
            run_backtest(df, train_bars=50, test_bars=40, n_hmm_states=2)

    def test_walk_forward_test_bars_too_small(self, df):
        with pytest.raises(InvalidParametersError, match="test_bars"):
            run_backtest(df, train_bars=120, test_bars=5, n_hmm_states=2)


class TestSimple:
    def test_simple_labelled_in_sample(self, simple_report):
        assert simple_report.in_sample is True
        text = simple_report.format_report()
        assert "IN-SAMPLE" in text
        assert "NOT an out-of-sample estimate" in text
        assert simple_report.to_dict()["sample"] == "in-sample"
        assert simple_report.windows == []

    def test_simple_covers_whole_period(self, df, simple_report):
        assert simple_report.n_bars == len(df)
        assert simple_report.start == df.index[0] and simple_report.end == df.index[-1]
        bh = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
        assert simple_report.buy_hold.total_return == pytest.approx(bh)
        assert simple_report.n_fits == 1

    def test_simple_metrics_consistent(self, simple_report):
        s = simple_report.strategy
        final = simple_report.equity_curve.iloc[-1]
        assert s.total_return == pytest.approx(final / simple_report.initial_capital - 1)
        assert s.n_trades == len(simple_report.trades)
        assert not math.isnan(s.sharpe)

    def test_simple_trades_frame_has_no_window(self, simple_report):
        assert "window" not in simple_report.trades_frame().columns

    def test_simple_insufficient_data(self, df):
        with pytest.raises(InsufficientDataError, match="Simple mode needs"):
            run_backtest(df.iloc[:50], mode="simple", n_hmm_states=2)

    def test_simple_detect_regimes_aligned(self, df):
        validator = WalkForwardValidator(strategy=RegimeStrategy(), n_hmm_states=2, hmm_n_iter=20)
        regimes, conf = validator.detect_regimes_in_sample(df)
        assert regimes.index.equals(df.index) and conf.index.equals(df.index)
        assert conf.between(0, 1).all()


class TestRunArgs:
    def test_run_unknown_mode(self, df):
        with pytest.raises(InvalidParametersError, match="Unknown mode"):
            run_backtest(df, mode="fancy")

    def test_run_bad_capital(self, df):
        with pytest.raises(InvalidParametersError, match="initial_capital"):
            run_backtest(df, initial_capital=0)

    def test_run_bad_timeframe(self, df):
        with pytest.raises(InvalidParametersError):
            run_backtest(df, timeframe="3W")

    def test_run_invalid_mapping_params(self, df):
        with pytest.raises(InvalidParametersError, match="unknown"):
            run_backtest(df, {"oops": 1})

    def test_run_empty_trades_frame_has_header(self, wf_report):
        empty = type(wf_report)(**{**wf_report.__dict__, "trades": []})
        frame = empty.trades_frame()
        assert frame.empty
        assert list(frame.columns) == ["window", *TRADE_COLUMNS]
