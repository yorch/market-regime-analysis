#!/usr/bin/env python3
"""
Jim Simons Market Regime Analysis System - Main Application (Click CLI)


This is the main entry point for the comprehensive market regime analysis system
implementing Jim Simons' Hidden Markov Model methodology for quantitative trading.
"""

import click

from market_regime_analysis import (
    MarketRegime,
    MarketRegimeAnalyzer,
    PortfolioHMMAnalyzer,
    SimonsRiskCalculator,
)


@click.group()
def cli() -> None:
    """Jim Simons Market Regime Analysis System CLI."""
    pass


@cli.command()
@click.option(
    "--provider",
    type=click.Choice(["yfinance", "alphavantage"]),
    default="alphavantage",
    help="Data provider",
)
@click.option("--api-key", type=str, required=False, help="Alpha Vantage API key")
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--timeframe",
    type=click.Choice(["1D", "1H", "15m"]),
    default="1D",
    help="Timeframe",
)
def detailed_analysis(provider: str, api_key: str, symbol: str, timeframe: str) -> None:
    """Run detailed HMM analysis for a single timeframe."""
    try:
        print(f"\nInitializing detailed analysis for {symbol} ({timeframe})...")
        if provider == "alphavantage" and not api_key:
            raise click.ClickException(
                "Alpha Vantage API key is required when using alphavantage provider"
            )

        analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider, api_key=api_key)
        analysis = analyzer.analyze_current_regime(timeframe)
        analyzer.print_analysis_report(timeframe)
        print("\n📋 DETAILED METRICS:")
        print(f"   HMM State: {analysis.hmm_state}/5")
        print(f"   Regime: {analysis.current_regime.value}")
        print(f"   Confidence: {analysis.regime_confidence:.3f}")
        print(f"   Persistence: {analysis.regime_persistence:.3f}")
        print(f"   Transition Prob: {analysis.transition_probability:.3f}")
        print("\n⚠️  RISK ASSESSMENT:")
        print(f"   Risk Level: {analysis.risk_level}")
        print(f"   Position Multiplier: {analysis.position_sizing_multiplier:.3f}")
        print(f"   Strategy: {analysis.recommended_strategy.value}")
    except Exception as e:
        print(f"Error: {e!s}")


@cli.command()
@click.option(
    "--provider",
    type=click.Choice(["yfinance", "alphavantage"]),
    default="alphavantage",
    help="Data provider",
)
@click.option("--api-key", type=str, required=False, help="Alpha Vantage API key")
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
def current_analysis(provider: str, api_key: str, symbol: str) -> None:
    """Run current HMM regime analysis for all timeframes."""
    try:
        print(f"\nInitializing analyzer for {symbol}...")
        if provider == "alphavantage" and not api_key:
            raise click.ClickException(
                "Alpha Vantage API key is required when using alphavantage provider"
            )

        analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider, api_key=api_key)
        for timeframe in ["1D", "1H", "15m"]:
            try:
                print(f"\n{'-' * 40}")
                print(f"ANALYSIS FOR {timeframe}")
                print(f"{'-' * 40}")
                analyzer.print_analysis_report(timeframe)
            except Exception as e:
                print(f"Error analyzing {timeframe}: {e!s}")
    except Exception as e:
        print(f"Error: {e!s}")


@cli.command()
@click.option(
    "--provider",
    type=click.Choice(["yfinance", "alphavantage"]),
    default="alphavantage",
    help="Data provider",
)
@click.option("--api-key", type=str, required=False, help="Alpha Vantage API key")
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--timeframe",
    type=click.Choice(["1D", "1H", "15m"]),
    default="1D",
    help="Timeframe",
)
@click.option("--days", type=int, default=60, help="Number of days to plot")
def generate_charts(provider: str, api_key: str, symbol: str, timeframe: str, days: int) -> None:
    """Generate HMM charts for a given symbol and timeframe."""
    try:
        print(f"Initializing analyzer for {symbol}...")
        if provider == "alphavantage" and not api_key:
            raise click.ClickException(
                "Alpha Vantage API key is required when using alphavantage provider"
            )

        analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider, api_key=api_key)
        print(f"Generating charts for {timeframe} ({days} days)...")
        analyzer.plot_regime_analysis(timeframe, days)
    except Exception as e:
        print(f"Error generating charts: {e!s}")


@cli.command()
@click.option(
    "--provider",
    type=click.Choice(["yfinance", "alphavantage"]),
    default="alphavantage",
    help="Data provider",
)
@click.option("--api-key", type=str, required=False, help="Alpha Vantage API key")
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option("--filename", type=str, default=None, help="Filename for CSV export")
def export_csv(provider: str, api_key: str, symbol: str, filename: str | None) -> None:
    """Export HMM analysis to CSV for a given symbol."""
    try:
        print(f"Initializing analyzer for {symbol}...")
        if provider == "alphavantage" and not api_key:
            raise click.ClickException(
                "Alpha Vantage API key is required when using alphavantage provider"
            )

        analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider, api_key=api_key)
        print("Exporting analysis data...")
        analyzer.export_analysis_to_csv(filename)
    except Exception as e:
        print(f"Error exporting data: {e!s}")


@cli.command()
@click.option("--base-size", type=float, default=0.02, help="Base position size (0.01-1.0)")
@click.option(
    "--regime",
    type=click.Choice([r.value for r in MarketRegime]),
    default=MarketRegime.BULL_TRENDING.value,
    help="Market regime",
)
@click.option("--confidence", type=float, default=0.8, help="Regime confidence (0.0-1.0)")
@click.option("--persistence", type=float, default=0.7, help="Regime persistence (0.0-1.0)")
@click.option("--correlation", type=float, default=0.0, help="Portfolio correlation (-1.0-1.0)")
def position_sizing(
    base_size: float,
    regime: str,
    confidence: float,
    persistence: float,
    correlation: float,
) -> None:
    """Calculate position sizing based on regime, confidence, persistence, and correlation."""
    try:
        regime_enum = next(r for r in MarketRegime if r.value == regime)
        size = SimonsRiskCalculator.calculate_regime_adjusted_size(
            base_size, regime_enum, confidence, persistence
        )
        correlation_adjusted = SimonsRiskCalculator.calculate_correlation_adjusted_size(
            size, correlation
        )
        print("\n📊 POSITION SIZING RESULTS:")
        print(f"   Base Size: {base_size:.1%}")
        print(f"   Regime: {regime}")
        print(f"   Regime Adjusted: {size:.1%}")
        print(f"   Correlation Adjusted: {correlation_adjusted:.1%}")
        print(f"   Final Recommendation: {correlation_adjusted:.1%}")
    except Exception as e:
        print(f"Error in position sizing calculation: {e!s}")


@cli.command()
@click.option(
    "--provider",
    type=click.Choice(["yfinance", "alphavantage"]),
    default="alphavantage",
    help="Data provider",
)
@click.option("--api-key", type=str, required=False, help="Alpha Vantage API key")
@click.option(
    "--symbols",
    type=str,
    default="SPY,QQQ,IWM",
    help="Comma-separated list of symbols (e.g., SPY,QQQ,IWM)",
)
@click.option(
    "--timeframe",
    type=click.Choice(["1D", "1H", "15m"]),
    default="1D",
    help="Timeframe",
)
def multi_symbol_analysis(provider: str, api_key: str, symbols: str, timeframe: str) -> None:
    """Run multi-symbol HMM analysis (portfolio)."""
    try:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        print(f"Initializing portfolio analysis for {len(symbol_list)} symbols...")
        if provider == "alphavantage":
            portfolio = PortfolioHMMAnalyzer(
                symbol_list, provider_flag=provider, api_key=api_key or ""
            )
        else:
            portfolio = PortfolioHMMAnalyzer(symbol_list, provider_flag=provider)
        portfolio.print_portfolio_summary(timeframe)
    except Exception as e:
        print(f"Error in portfolio analysis: {e!s}")


@cli.command()
@click.option(
    "--provider",
    type=click.Choice(["yfinance", "alphavantage"]),
    default="alphavantage",
    help="Data provider",
)
@click.option("--api-key", type=str, required=False, help="Alpha Vantage API key")
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--interval",
    type=int,
    default=300,
    help="Refresh interval in seconds (default: 300)",
)
def continuous_monitoring(provider: str, api_key: str, symbol: str, interval: int) -> None:
    """Start continuous HMM monitoring for a symbol."""
    try:
        print(f"Starting continuous monitoring for {symbol}...")
        if provider == "alphavantage" and not api_key:
            raise click.ClickException(
                "Alpha Vantage API key is required when using alphavantage provider"
            )

        analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider, api_key=api_key)
        analyzer.run_continuous_monitoring(interval)
    except Exception as e:
        print(f"Error in continuous monitoring: {e!s}")


if __name__ == "__main__":
    cli()
