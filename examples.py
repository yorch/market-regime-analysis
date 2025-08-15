#!/usr/bin/env python3
"""
Example usage of the Market Regime Analysis System.

This file demonstrates the basic usage patterns described in PLAN.md.
"""

import os
import sys

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_regime_analysis import (
    MarketRegime,
    MarketRegimeAnalyzer,
    PortfolioHMMAnalyzer,
    SimonsRiskCalculator,
)


def example_single_symbol_analysis():
    """Example 1: Single symbol analysis as described in PLAN.md."""
    print("=" * 60)
    print("EXAMPLE 1: SINGLE SYMBOL ANALYSIS")
    print("=" * 60)

    try:
        # Single symbol analysis
        analyzer = MarketRegimeAnalyzer("SPY")
        analysis = analyzer.analyze_current_regime("1D")
        analyzer.print_analysis_report("1D")

        print("\nKey Results:")
        print(f"Regime: {analysis.current_regime.value}")
        print(f"Strategy: {analysis.recommended_strategy.value}")
        print(f"Position Size: {analysis.position_sizing_multiplier:.1%}")

    except Exception as e:
        print(f"Error: {e!s}")
        print("Note: This requires internet connection for Yahoo Finance data")


def example_portfolio_analysis():
    """Example 2: Portfolio analysis as described in PLAN.md."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: PORTFOLIO ANALYSIS")
    print("=" * 60)

    try:
        # Portfolio analysis
        portfolio = PortfolioHMMAnalyzer(["SPY", "QQQ", "IWM"])
        portfolio.print_portfolio_summary()

    except Exception as e:
        print(f"Error: {e!s}")
        print("Note: This requires internet connection for Yahoo Finance data")


def example_risk_calculation():
    """Example 3: Risk calculation as described in PLAN.md."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: RISK CALCULATION")
    print("=" * 60)

    try:
        # Example regime and parameters
        regime = MarketRegime.BULL_TRENDING
        confidence = 0.85
        persistence = 0.70
        base_size = 0.02

        # Risk calculation
        size = SimonsRiskCalculator.calculate_regime_adjusted_size(
            base_size=base_size,
            regime=regime,
            confidence=confidence,
            persistence=persistence,
        )

        print("Risk Calculation Example:")
        print(f"Base Size: {base_size:.1%}")
        print(f"Regime: {regime.value}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Persistence: {persistence:.1%}")
        print(f"Adjusted Size: {size:.1%}")

        # Kelly Criterion example
        kelly_size = SimonsRiskCalculator.calculate_kelly_optimal_size(
            win_rate=0.55, avg_win=0.025, avg_loss=0.020, confidence=0.8
        )

        print("\nKelly Criterion Example:")
        print("Win Rate: 55%")
        print("Average Win: 2.5%")
        print("Average Loss: 2.0%")
        print(f"Kelly Optimal Size: {kelly_size:.1%}")

    except Exception as e:
        print(f"Error: {e!s}")


def example_comprehensive_analysis():
    """Example 4: Comprehensive analysis workflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: COMPREHENSIVE ANALYSIS WORKFLOW")
    print("=" * 60)

    try:
        # Initialize analyzer
        symbol = "SPY"
        print(f"Analyzing {symbol} across multiple timeframes...")

        analyzer = MarketRegimeAnalyzer(symbol)

        # Analyze each timeframe
        for timeframe in ["1D", "1H", "15m"]:
            try:
                analysis = analyzer.analyze_current_regime(timeframe)

                print(f"\n{timeframe} Analysis:")
                print(f"  Regime: {analysis.current_regime.value}")
                print(f"  Confidence: {analysis.regime_confidence:.1%}")
                print(f"  Strategy: {analysis.recommended_strategy.value}")
                print(f"  Position Size: {analysis.position_sizing_multiplier:.1%}")
                print(f"  Risk Level: {analysis.risk_level}")

                if analysis.arbitrage_opportunities:
                    print(f"  Arbitrage Opportunities: {len(analysis.arbitrage_opportunities)}")

            except Exception as e:
                print(f"  Error in {timeframe}: {e!s}")

        # Export analysis
        print("\nExporting analysis to CSV...")
        analyzer.export_analysis_to_csv()

    except Exception as e:
        print(f"Error: {e!s}")
        print("Note: This requires internet connection for Yahoo Finance data")


def main():
    """Main function demonstrating all examples."""
    print("JIM SIMONS MARKET REGIME ANALYSIS SYSTEM")
    print("Usage Examples from PLAN.md")
    print("=" * 80)

    # Run examples
    example_single_symbol_analysis()
    example_portfolio_analysis()
    example_risk_calculation()
    example_comprehensive_analysis()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nTo run the interactive application:")
    print("python main.py")
    print("\nTo test core functionality without internet:")
    print("python test_mock.py")


if __name__ == "__main__":
    main()
