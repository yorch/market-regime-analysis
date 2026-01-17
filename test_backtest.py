"""
Test backtesting framework with regime-based strategy.

This script validates the new backtesting capabilities by:
1. Running a regime-based trading strategy on historical data
2. Calculating realistic transaction costs
3. Computing comprehensive performance metrics
4. Generating Kelly Criterion parameters from actual trades
"""

import sys

import pandas as pd

from market_regime_analysis.backtester import BacktestEngine, EquityCostModel
from market_regime_analysis.enums import MarketRegime, TradingStrategy
from market_regime_analysis.providers import MarketDataProvider
from market_regime_analysis.true_hmm_detector import TrueHMMDetector


def create_simple_regime_strategy(regimes: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Create simple trading strategy from regimes.

    Args:
        regimes: Series of MarketRegime values

    Returns:
        Tuple of (strategies, position_sizes)
    """
    strategies = []
    position_sizes = []

    for regime in regimes:
        # Map regimes to strategies and position sizes
        if regime == MarketRegime.BULL_TRENDING:
            strategies.append(TradingStrategy.TREND_FOLLOWING)
            position_sizes.append(1.3)  # Aggressive in bull trends
        elif regime == MarketRegime.BEAR_TRENDING:
            strategies.append(TradingStrategy.DEFENSIVE)
            position_sizes.append(0.3)  # Defensive in bear
        elif regime == MarketRegime.MEAN_REVERTING:
            strategies.append(TradingStrategy.MEAN_REVERSION)
            position_sizes.append(1.2)
        elif regime == MarketRegime.LOW_VOLATILITY:
            strategies.append(TradingStrategy.MOMENTUM)
            position_sizes.append(1.1)
        elif regime == MarketRegime.HIGH_VOLATILITY:
            strategies.append(TradingStrategy.DEFENSIVE)
            position_sizes.append(0.4)  # Reduce exposure
        elif regime == MarketRegime.BREAKOUT:
            strategies.append(TradingStrategy.MOMENTUM)
            position_sizes.append(0.9)
        else:  # UNKNOWN
            strategies.append(TradingStrategy.AVOID)
            position_sizes.append(0.2)

    return pd.Series(strategies, index=regimes.index), pd.Series(position_sizes, index=regimes.index)


def main():
    """Run comprehensive backtest."""
    print("=" * 100)
    print("REGIME-BASED STRATEGY BACKTESTING")
    print("=" * 100)

    # 1. Load historical data
    print("\n1. Loading historical data...")
    try:
        provider = MarketDataProvider.create_provider("yfinance")
        df = provider.fetch("SPY", "2y", "1d")
        print(f"   ✓ Loaded {len(df)} days of SPY data")
        print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    except Exception as e:
        print(f"   ✗ Failed to load data: {e}")
        sys.exit(1)

    # 2. Train HMM and detect regimes (WITH PROPER WALK-FORWARD)
    print("\n2. Training HMM and detecting regimes...")
    print("   ⚠️  Using walk-forward approach to avoid look-ahead bias")
    try:
        # Get regime predictions for each day using ONLY past data
        regime_list = []
        state_list = []
        confidence_list = []

        # Minimum training period
        min_train_days = 100
        retrain_frequency = 20  # Retrain every 20 days

        print(f"   Training window: {min_train_days} days minimum")
        print(f"   Retrain frequency: Every {retrain_frequency} days")

        hmm = None
        for i in range(min_train_days, len(df)):
            # Retrain periodically or on first iteration
            if i == min_train_days or (i - min_train_days) % retrain_frequency == 0:
                # Train on data UP TO (but not including) current day
                train_df = df.iloc[:i]
                hmm = TrueHMMDetector(n_states=6, n_iter=100)
                hmm.fit(train_df)

            # Predict using data UP TO current day (NOT including current close)
            # In reality, we'd use yesterday's close to make today's trading decision
            predict_df = df.iloc[:i]  # Only past data
            regime, state, conf = hmm.predict_regime(predict_df, use_viterbi=False)

            regime_list.append(regime)
            state_list.append(state)
            confidence_list.append(conf)

        # Create regime series aligned with price data
        regime_series = pd.Series(
            regime_list,
            index=df.index[min_train_days:]
        )

        # Align data
        df_backtest = df.iloc[min_train_days:].copy()

        print(f"   ✓ Detected regimes for {len(regime_series)} days (no look-ahead bias)")
        print(f"\n   Regime Distribution:")
        regime_counts = regime_series.value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(regime_series) * 100
            print(f"     {regime.value:.<30} {count:>4} ({pct:>5.1f}%)")

    except Exception as e:
        print(f"   ✗ Regime detection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Create trading strategy from regimes
    print("\n3. Creating trading strategy...")
    try:
        strategies, position_sizes = create_simple_regime_strategy(regime_series)
        print(f"   ✓ Strategy created for {len(strategies)} days")
        print(f"\n   Strategy Distribution:")
        strategy_counts = strategies.value_counts()
        for strategy, count in strategy_counts.items():
            pct = count / len(strategies) * 100
            print(f"     {strategy.value:.<30} {count:>4} ({pct:>5.1f}%)")
    except Exception as e:
        print(f"   ✗ Strategy creation failed: {e}")
        sys.exit(1)

    # 4. Run backtest
    print("\n4. Running backtest...")
    try:
        engine = BacktestEngine(
            initial_capital=100000.0,
            cost_model=EquityCostModel(),
            max_position_size=0.20,  # 20% max position
            stop_loss_pct=0.10,  # 10% stop loss
            take_profit_pct=None,  # No take profit
        )

        results = engine.run_regime_strategy(
            df=df_backtest,
            regimes=regime_series,
            strategies=strategies,
            position_sizes=position_sizes,
        )

        print(f"   ✓ Backtest complete")
        print(f"     - Total trades: {len(results['trades'])}")
        print(f"     - Final capital: ${results['final_capital']:,.2f}")
        print(f"     - Total return: {results['total_return']:.2%}")

    except Exception as e:
        print(f"   ✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. Print detailed results
    print("\n5. Performance Analysis...")
    engine.print_results(results)

    # 6. Key findings
    print("\n6. Key Findings:")
    perf = results["performance"].metrics

    print("\n   📊 PROFITABILITY CHECK:")
    is_profitable = results["performance"].is_profitable(min_sharpe=0.5, min_trades=30)
    if is_profitable:
        print("   ✅ Strategy is profitable - meets deployment criteria")
    else:
        print("   ❌ Strategy is NOT profitable - do not deploy")

    print("\n   🎯 KELLY CRITERION (FROM BACKTEST):")
    print(f"     - Win Rate:         {perf['win_rate']:.2%}")
    print(f"     - Avg Win:          ${perf['avg_win']:.2f}")
    print(f"     - Avg Loss:         ${perf['avg_loss']:.2f}")
    print(f"     - Full Kelly:       {perf['kelly_fraction']:.2%}")
    print(f"     - Half Kelly:       {perf['half_kelly']:.2%}")
    print(f"     - Quarter Kelly:    {perf['quarter_kelly']:.2%}")

    print("\n   💡 INSIGHTS:")
    print(f"     - Sharpe Ratio > 1.0: {'✅ YES' if perf['sharpe_ratio'] > 1.0 else '❌ NO'}")
    print(f"     - Max Drawdown < 20%: {'✅ YES' if perf['max_drawdown'] > -0.20 else '❌ NO'}")
    print(f"     - Profit Factor > 1.5: {'✅ YES' if perf['profit_factor'] > 1.5 else '❌ NO'}")
    print(f"     - Win Rate > 50%: {'✅ YES' if perf['win_rate'] > 0.5 else '❌ NO'}")

    # 7. Compare with buy-and-hold
    print("\n7. Benchmark Comparison:")
    buy_hold_return = (df_backtest['Close'].iloc[-1] / df_backtest['Close'].iloc[0] - 1)
    strategy_return = results['total_return']
    print(f"   Buy & Hold:    {buy_hold_return:>10.2%}")
    print(f"   This Strategy: {strategy_return:>10.2%}")
    print(f"   Difference:    {strategy_return - buy_hold_return:>10.2%}")

    if strategy_return > buy_hold_return:
        print("   ✅ Strategy outperforms buy-and-hold")
    else:
        print("   ❌ Strategy underperforms buy-and-hold")

    print("\n" + "=" * 100)
    print("BACKTEST COMPLETE")
    print("=" * 100)

    print("\n💬 CONCLUSION:")
    if is_profitable and strategy_return > 0:
        print("   The regime-based strategy shows promise in backtesting.")
        print("   Next steps:")
        print("   1. Run walk-forward analysis for robustness")
        print("   2. Test on other symbols (QQQ, IWM, etc.)")
        print("   3. Paper trade for 3-6 months before live deployment")
        print("   4. Start with small capital (<$5k) if live trading")
    else:
        print("   The strategy does NOT meet profitability criteria.")
        print("   Do NOT deploy with real capital.")
        print("   Consider:")
        print("   1. Adjusting regime classification thresholds")
        print("   2. Different position sizing rules")
        print("   3. Alternative entry/exit logic")
        print("   4. Transaction cost reduction strategies")


if __name__ == "__main__":
    main()
