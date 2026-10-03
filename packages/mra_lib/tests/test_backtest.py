"""Regime-driven backtest on deterministic synthetic data (walk-forward, no look-ahead)."""

import pandas as pd
import pytest

from mra_lib.backtesting import BacktestEngine, EquityCostModel
from mra_lib.config.enums import MarketRegime, TradingStrategy
from mra_lib.indicators.true_hmm_detector import TrueHMMDetector

REGIME_TO_STRATEGY: dict[MarketRegime, tuple[TradingStrategy, float]] = {
    MarketRegime.BULL_TRENDING: (TradingStrategy.TREND_FOLLOWING, 1.3),
    MarketRegime.BEAR_TRENDING: (TradingStrategy.DEFENSIVE, 0.3),
    MarketRegime.MEAN_REVERTING: (TradingStrategy.MEAN_REVERSION, 1.2),
    MarketRegime.LOW_VOLATILITY: (TradingStrategy.MOMENTUM, 1.1),
    MarketRegime.HIGH_VOLATILITY: (TradingStrategy.DEFENSIVE, 0.4),
    MarketRegime.BREAKOUT: (TradingStrategy.MOMENTUM, 0.9),
}


def regimes_to_strategy(regimes: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Map each regime to a (strategy, position-size multiplier) pair."""
    pairs = [REGIME_TO_STRATEGY.get(r, (TradingStrategy.AVOID, 0.2)) for r in regimes]
    return (
        pd.Series([p[0] for p in pairs], index=regimes.index),
        pd.Series([p[1] for p in pairs], index=regimes.index),
    )


def walk_forward_regimes(df: pd.DataFrame, min_train: int, retrain_every: int) -> pd.Series:
    """Predict a regime for each bar using only data strictly before it."""
    regimes = []
    hmm: TrueHMMDetector | None = None
    for i in range(min_train, len(df)):
        if hmm is None or (i - min_train) % retrain_every == 0:
            hmm = TrueHMMDetector(n_states=4, n_iter=30).fit(df.iloc[:i])
        regime, _, _ = hmm.predict_regime(df.iloc[:i], use_viterbi=False)
        regimes.append(regime)
    return pd.Series(regimes, index=df.index[min_train:])


@pytest.fixture(scope="module")
def backtest_result(synthetic_ohlcv):
    df = synthetic_ohlcv(n=260)
    regimes = walk_forward_regimes(df, min_train=150, retrain_every=40)
    strategies, sizes = regimes_to_strategy(regimes)
    engine = BacktestEngine(
        initial_capital=100_000.0,
        cost_model=EquityCostModel(),
        max_position_size=0.20,
        stop_loss_pct=0.10,
        take_profit_pct=None,
    )
    results = engine.run_regime_strategy(
        df=df.iloc[150:], regimes=regimes, strategies=strategies, position_sizes=sizes
    )
    return regimes, strategies, results


def test_walk_forward_regimes_cover_every_test_bar(backtest_result):
    regimes, strategies, _ = backtest_result

    assert len(regimes) == 110
    assert all(isinstance(r, MarketRegime) for r in regimes)
    assert all(isinstance(s, TradingStrategy) for s in strategies)


def test_regime_backtest_results_are_consistent(backtest_result):
    _, _, results = backtest_result

    assert set(results) >= {"trades", "equity_curve", "performance", "final_capital"}
    assert results["final_capital"] > 0
    assert results["total_return"] == pytest.approx(results["final_capital"] / 100_000.0 - 1)
    metrics = results["performance"].metrics
    for key in ("win_rate", "sharpe_ratio", "max_drawdown", "kelly_fraction"):
        assert key in metrics
    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert metrics["max_drawdown"] <= 0.0


def test_regime_to_strategy_mapping_defaults_to_avoid():
    regimes = pd.Series([MarketRegime.UNKNOWN, MarketRegime.BULL_TRENDING])

    strategies, sizes = regimes_to_strategy(regimes)

    assert list(strategies) == [TradingStrategy.AVOID, TradingStrategy.TREND_FOLLOWING]
    assert list(sizes) == [0.2, 1.3]
