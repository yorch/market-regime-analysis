#!/usr/bin/env python3
"""
Jim Simons Market Regime Analysis System - Main Application (Click CLI)


This is the main entry point for the comprehensive market regime analysis system
implementing Jim Simons' Hidden Markov Model methodology for quantitative trading.
"""

import concurrent.futures
import functools
import os
from typing import Any

import click

from market_regime_analysis import (
    MarketRegime,
    MarketRegimeAnalyzer,
    PortfolioHMMAnalyzer,
    SimonsRiskCalculator,
)
from market_regime_analysis.providers import MarketDataProvider


def validate_api_key(provider: str, api_key: str | None) -> str:
    """Validate and retrieve API key for providers that require it.

    Args:
        provider: Data provider name
        api_key: Optional API key from command line

    Returns:
        str: Valid API key

    Raises:
        click.ClickException: If required API key is missing
    """
    if provider not in ["alphavantage", "polygon"]:
        return api_key or ""

    if api_key:
        return api_key

    # Try environment variables
    env_keys = {
        "alphavantage": ["ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY"],
        "polygon": ["POLYGON_API_KEY"],
    }

    for env_var in env_keys.get(provider, []):
        api_key = os.getenv(env_var)
        if api_key:
            return api_key

    provider_name = provider.replace("_", " ").title()
    raise click.ClickException(
        f"{provider_name} API key is required when using {provider} provider. "
        f"Set {env_keys[provider][0]} environment variable or use --api-key option."
    )


def validate_percentage(ctx: click.Context, param: click.Parameter, value: float) -> float:
    """Validate percentage values (0.0-1.0)."""
    _ = ctx, param  # Suppress unused parameter warnings
    if not 0.0 <= value <= 1.0:
        raise click.BadParameter("Must be between 0.0 and 1.0")
    return value


def validate_correlation(ctx: click.Context, param: click.Parameter, value: float) -> float:
    """Validate correlation values (-1.0-1.0)."""
    _ = ctx, param  # Suppress unused parameter warnings
    if not -1.0 <= value <= 1.0:
        raise click.BadParameter("Must be between -1.0 and 1.0")
    return value


def validate_positive_int(ctx: click.Context, param: click.Parameter, value: int) -> int:
    """Validate positive integer values."""
    _ = ctx, param  # Suppress unused parameter warnings
    if value <= 0:
        raise click.BadParameter("Must be a positive integer")
    return value


def handle_exceptions(func):
    """Decorator for consistent exception handling across CLI commands."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except click.ClickException:
            raise  # Re-raise Click exceptions
        except ValueError as e:
            click.echo(f"❌ Invalid input: {e}", err=True)
            raise click.Abort() from e
        except ConnectionError as e:
            click.echo(f"🌐 Network error: {e}", err=True)
            raise click.Abort() from e
        except FileNotFoundError as e:
            click.echo(f"📁 File not found: {e}", err=True)
            raise click.Abort() from e
        except PermissionError as e:
            click.echo(f"🔒 Permission error: {e}", err=True)
            raise click.Abort() from e
        except Exception as e:
            click.echo(f"💥 Unexpected error: {e}", err=True)
            raise click.Abort() from e

    return wrapper


def analyze_timeframe_parallel(
    analyzer: MarketRegimeAnalyzer, timeframe: str
) -> tuple[str, Any | Exception]:
    """Analyze a single timeframe for parallel processing.

    Args:
        analyzer: Market regime analyzer instance
        timeframe: Timeframe to analyze

    Returns:
        Tuple of (timeframe, result_or_exception)
    """
    try:
        analysis = analyzer.analyze_current_regime(timeframe)
        return timeframe, analysis
    except Exception as e:
        return timeframe, e


@click.group()
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
@click.option(
    "--provider",
    type=click.Choice(["yfinance", "alphavantage", "polygon"]),
    default="alphavantage",
    help="Data provider for all commands",
)
@click.option(
    "--api-key",
    type=str,
    help="API key (SECURITY WARNING: prefer environment variables over CLI arguments)",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool, provider: str, api_key: str | None) -> None:
    """Jim Simons Market Regime Analysis System CLI."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["provider"] = provider
    ctx.obj["api_key"] = validate_api_key(provider, api_key)


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--timeframe",
    type=click.Choice(["1D", "1H", "15m"]),
    default="1D",
    help="Timeframe",
)
@click.pass_context
@handle_exceptions
def detailed_analysis(ctx: click.Context, symbol: str, timeframe: str) -> None:
    """Run detailed HMM analysis for a single timeframe.

    Args:
        ctx: Click context containing global options
        symbol: Trading symbol to analyze (e.g., 'SPY', 'QQQ')
        timeframe: Time interval for analysis ('1D', '1H', '15m')
    """
    print(f"\nInitializing detailed analysis for {symbol} ({timeframe})...")

    # Get provider and API key from context
    provider = ctx.obj["provider"]
    api_key = ctx.obj["api_key"]

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


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.pass_context
@handle_exceptions
def current_analysis(ctx: click.Context, symbol: str) -> None:
    """Run current HMM regime analysis for all timeframes.

    Args:
        ctx: Click context containing global options
        symbol: Trading symbol to analyze (e.g., 'SPY', 'QQQ')
    """
    print(f"\nInitializing analyzer for {symbol}...")

    # Get provider and API key from context
    provider = ctx.obj["provider"]
    api_key = ctx.obj["api_key"]

    analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider, api_key=api_key)
    timeframes = ["1D", "1H", "15m"]

    # Use parallel processing for independent timeframe analyses
    print("🔄 Analyzing timeframes in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all timeframe analyses
        future_to_timeframe = {
            executor.submit(analyze_timeframe_parallel, analyzer, tf): tf for tf in timeframes
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_timeframe):
            timeframe, result = future.result()

            print(f"\n{'-' * 40}")
            print(f"ANALYSIS FOR {timeframe}")
            print(f"{'-' * 40}")

            if isinstance(result, Exception):
                click.echo(f"❌ Error analyzing {timeframe}: {result}", err=True)
            else:
                analyzer.print_analysis_report(timeframe)


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--timeframe",
    type=click.Choice(["1D", "1H", "15m"]),
    default="1D",
    help="Timeframe",
)
@click.option(
    "--days", type=int, default=60, callback=validate_positive_int, help="Number of days to plot"
)
@click.pass_context
@handle_exceptions
def generate_charts(ctx: click.Context, symbol: str, timeframe: str, days: int) -> None:
    """Generate HMM charts for a given symbol and timeframe.

    Args:
        ctx: Click context containing global options
        symbol: Trading symbol to analyze (e.g., 'SPY', 'QQQ')
        timeframe: Time interval for analysis ('1D', '1H', '15m')
        days: Number of days to plot (must be positive)
    """
    print(f"Initializing analyzer for {symbol}...")

    # Get provider and API key from context
    provider = ctx.obj["provider"]
    api_key = ctx.obj["api_key"]

    analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider, api_key=api_key)
    print(f"Generating charts for {timeframe} ({days} days)...")
    analyzer.plot_regime_analysis(timeframe, days)


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option("--filename", type=str, default=None, help="Filename for CSV export")
@click.pass_context
@handle_exceptions
def export_csv(ctx: click.Context, symbol: str, filename: str | None) -> None:
    """Export HMM analysis to CSV for a given symbol.

    Args:
        ctx: Click context containing global options
        symbol: Trading symbol to analyze (e.g., 'SPY', 'QQQ')
        filename: Optional filename for CSV export
    """
    print(f"Initializing analyzer for {symbol}...")

    # Get provider and API key from context
    provider = ctx.obj["provider"]
    api_key = ctx.obj["api_key"]

    analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider, api_key=api_key)
    print("Exporting analysis data...")
    analyzer.export_analysis_to_csv(filename)


@cli.command()
def list_providers() -> None:
    """List all available data providers and their capabilities."""
    providers = MarketDataProvider.get_available_providers()

    print("\n📡 AVAILABLE DATA PROVIDERS:")
    print("=" * 50)

    for name, info in providers.items():
        print(f"\n🔹 {name.upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Requires API Key: {'Yes' if info['requires_api_key'] else 'No'}")
        print(f"   Rate Limit: {info['rate_limit_per_minute']} req/min")
        print(f"   Supported Intervals: {', '.join(sorted(info['supported_intervals']))}")
        print(f"   Supported Periods: {', '.join(sorted(info['supported_periods']))}")

    print("\n💡 USAGE EXAMPLES:")
    print("   --provider yfinance")
    print("   --provider alphavantage --api-key YOUR_KEY")
    print("   --provider polygon --api-key YOUR_KEY")
    print("   export ALPHA_VANTAGE_API_KEY=your_key")
    print("   export POLYGON_API_KEY=your_key")


@cli.command()
@click.option(
    "--base-size",
    type=float,
    default=0.02,
    callback=validate_percentage,
    help="Base position size (0.0-1.0)",
)
@click.option(
    "--regime",
    type=click.Choice([r.value for r in MarketRegime]),
    default=MarketRegime.BULL_TRENDING.value,
    help="Market regime",
)
@click.option(
    "--confidence",
    type=float,
    default=0.8,
    callback=validate_percentage,
    help="Regime confidence (0.0-1.0)",
)
@click.option(
    "--persistence",
    type=float,
    default=0.7,
    callback=validate_percentage,
    help="Regime persistence (0.0-1.0)",
)
@click.option(
    "--correlation",
    type=float,
    default=0.0,
    callback=validate_correlation,
    help="Portfolio correlation (-1.0-1.0)",
)
@handle_exceptions
def position_sizing(
    base_size: float,
    regime: str,
    confidence: float,
    persistence: float,
    correlation: float,
) -> None:
    """Calculate position sizing based on regime, confidence, persistence, and correlation.

    Args:
        base_size: Base position size (0.0-1.0)
        regime: Market regime type
        confidence: Regime confidence level (0.0-1.0)
        persistence: Regime persistence level (0.0-1.0)
        correlation: Portfolio correlation (-1.0-1.0)
    """
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


@cli.command()
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
@click.pass_context
@handle_exceptions
def multi_symbol_analysis(ctx: click.Context, symbols: str, timeframe: str) -> None:
    """Run multi-symbol HMM analysis (portfolio).

    Args:
        ctx: Click context containing global options
        symbols: Comma-separated list of symbols to analyze
        timeframe: Time interval for analysis ('1D', '1H', '15m')
    """
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise click.BadParameter("At least one symbol must be provided")

    print(f"Initializing portfolio analysis for {len(symbol_list)} symbols...")

    # Get provider and API key from context
    provider = ctx.obj["provider"]
    api_key = ctx.obj["api_key"]

    portfolio = PortfolioHMMAnalyzer(symbol_list, provider_flag=provider, api_key=api_key)
    portfolio.print_portfolio_summary(timeframe)


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--interval",
    type=int,
    default=300,
    callback=validate_positive_int,
    help="Refresh interval in seconds (default: 300)",
)
@click.pass_context
@handle_exceptions
def continuous_monitoring(ctx: click.Context, symbol: str, interval: int) -> None:
    """Start continuous HMM monitoring for a symbol.

    Args:
        ctx: Click context containing global options
        symbol: Trading symbol to monitor (e.g., 'SPY', 'QQQ')
        interval: Refresh interval in seconds (must be positive)
    """
    print(f"Starting continuous monitoring for {symbol}...")

    # Get provider and API key from context
    provider = ctx.obj["provider"]
    api_key = ctx.obj["api_key"]

    analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider, api_key=api_key)
    analyzer.run_continuous_monitoring(interval)


if __name__ == "__main__":
    cli()
