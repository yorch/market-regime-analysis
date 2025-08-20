#!/usr/bin/env python3
"""
Curated programmatic usage examples for Market Regime Analysis.

This consolidates and de-duplicates the previous examples.py and
examples_programmatic_usage.py into a single, focused script.
"""

from __future__ import annotations

import os

from examples.common import Banner
from market_regime_analysis import (
    MarketRegime,
    MarketRegimeAnalyzer,
    PortfolioHMMAnalyzer,
    SimonsRiskCalculator,
)
from market_regime_analysis.providers import list_available_providers

banner = Banner()


def single_asset_analysis() -> None:
    banner.title("Example 1: Single Asset Regime Analysis")

    analyzer = MarketRegimeAnalyzer(
        symbol="SPY",
        periods={"1D": "1y", "1H": "3mo"},
        provider_flag="yfinance",
    )

    for timeframe in analyzer.periods:
        analysis = analyzer.analyze_current_regime(timeframe)
        print(f"\n{timeframe} Analysis:")
        print(f"  Regime: {analysis.current_regime.value}")
        print(f"  Strategy: {analysis.recommended_strategy.value}")
        print(f"  Confidence: {analysis.regime_confidence:.1%}")
        print(f"  Position Size: {analysis.position_sizing_multiplier:.1%}")
        print(f"  Risk Level: {analysis.risk_level}")
        latest_price = analyzer.data[timeframe]["Close"].iloc[-1]
        print(f"  Latest Price: ${latest_price:.2f}")


def portfolio_analysis() -> None:
    banner.title("Example 2: Portfolio Regime Analysis")

    symbols = ["SPY", "QQQ", "IWM"]
    portfolio = PortfolioHMMAnalyzer(
        symbols=symbols, periods={"1D": "6mo"}, provider_flag="yfinance"
    )

    for symbol in symbols:
        analyzer = portfolio.analyzers.get(symbol)
        if not analyzer:
            continue
        analysis = analyzer.analyze_current_regime("1D")
        print(f"\n{symbol}:")
        print(f"  Regime: {analysis.current_regime.value}")
        print(f"  Confidence: {analysis.regime_confidence:.1%}")
        print(f"  Strategy: {analysis.recommended_strategy.value}")


def risk_management() -> None:
    banner.title("Example 3: Risk Management & Position Sizing")

    base_size = 0.1
    kelly_size = SimonsRiskCalculator.calculate_kelly_optimal_size(
        win_rate=0.65, avg_win=2.5, avg_loss=1.0, confidence=0.8
    )
    print(f"Base Size: {base_size:.1%}")
    print(f"Kelly Optimal Size: {kelly_size:.1%}")

    for regime in (
        MarketRegime.BULL_TRENDING,
        MarketRegime.HIGH_VOLATILITY,
        MarketRegime.MEAN_REVERTING,
    ):
        adjusted = SimonsRiskCalculator.calculate_regime_adjusted_size(
            base_size=kelly_size, regime=regime, confidence=0.8, persistence=0.7
        )
        print(f"  {regime.value}: {adjusted:.1%}")

    comprehensive = SimonsRiskCalculator.calculate_comprehensive_position_size(
        base_size=base_size,
        regime=MarketRegime.BULL_TRENDING,
        confidence=0.8,
        persistence=0.7,
        correlation=0.3,
        win_rate=0.65,
        avg_win=2.5,
        avg_loss=1.0,
        current_vol=0.20,
        historical_vol=0.16,
    )
    print("\nComprehensive Position Sizing:")
    for k, v in comprehensive.items():
        print(f"  {k}: {v:.3f}")


def provider_overview() -> None:
    banner.title("Example 4: Data Provider Overview")

    providers = list_available_providers()
    for name, info in providers.items():
        print(f"  • {name}")
        print(f"    Description: {info['description']}")
        print(f"    API Key Required: {info['requires_api_key']}")
        print(f"    Rate Limit: {info['rate_limit_per_minute']}/min")


def integration_patterns() -> None:
    banner.title("Example 5: Integration Patterns")

    analyzer = MarketRegimeAnalyzer(symbol="SPY", periods={"1D": "6mo"})
    analysis = analyzer.analyze_current_regime("1D")

    mapping = {
        MarketRegime.BULL_TRENDING: "momentum_long",
        MarketRegime.BEAR_TRENDING: "momentum_short",
        MarketRegime.MEAN_REVERTING: "mean_reversion",
        MarketRegime.HIGH_VOLATILITY: "volatility_trading",
        MarketRegime.LOW_VOLATILITY: "trend_following",
        MarketRegime.BREAKOUT: "breakout_trading",
        MarketRegime.UNKNOWN: "risk_off",
    }

    strategy = mapping.get(analysis.current_regime, "risk_off")
    print(f"Regime: {analysis.current_regime.value}")
    print(f"Recommended Strategy: {strategy}")

    # Export to CSV then clean up
    fname = "regime_analysis.csv"
    analyzer.export_analysis_to_csv(fname)
    print(f"Exported to {fname}")
    if os.path.exists(fname):
        os.remove(fname)


def main() -> None:
    print("🚀 Market Regime Analysis - Programmatic Examples")
    print("=" * 60)

    examples = [
        ("Single Asset", single_asset_analysis),
        ("Portfolio", portfolio_analysis),
        ("Risk Management", risk_management),
        ("Providers", provider_overview),
        ("Integration", integration_patterns),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:  # pragma: no cover - examples should not crash CI
            print(f"✖ Error in {name}: {e}")
            continue

    print("\nAll examples completed.")


if __name__ == "__main__":
    main()
