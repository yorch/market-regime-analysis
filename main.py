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
from market_regime_analysis.backtester import RegimeMultiplierCalibrator
from market_regime_analysis.providers import MarketDataProvider
from market_regime_analysis.true_hmm_detector import TrueHMMDetector


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
    """
    Jim Simons Market Regime Analysis System CLI.

    This system implements Hidden Markov Model methodology for quantitative trading analysis,
    detecting market regimes and providing statistical arbitrage signals.

    🌐 WEB API AVAILABLE: Run 'python start_api.py --dev' to start the REST API server
    📚 API Documentation: http://localhost:8000/docs (when API server is running)
    🔌 WebSocket Monitoring: ws://localhost:8000/ws/monitoring/{symbol}

    For programmatic access, see examples_api.py for Python client usage.
    """
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


@cli.command()
@click.option("--host", default="0.0.0.0", help="API server host")
@click.option("--port", default=8000, help="API server port")
@click.option("--dev/--no-dev", default=False, help="Development mode with auto-reload")
@handle_exceptions
def start_api(host: str, port: int, dev: bool) -> None:
    """Start the REST API server for web access.

    This command starts the FastAPI server providing REST API endpoints
    for all CLI functionality plus WebSocket monitoring.

    Examples:
        uv run main.py start-api --dev              # Development mode
        uv run main.py start-api --host 0.0.0.0     # Production mode
    """
    try:
        import uvicorn  # noqa: PLC0415

        # Validate api_server module is available
        import api_server  # noqa: F401, PLC0415

        print("🚀 Starting Market Regime Analysis API Server")
        print(f"🌐 Host: {host}")
        print(f"🔌 Port: {port}")
        print(f"🔄 Development mode: {dev}")

        if dev:
            print(f"📖 API Documentation: http://{host}:{port}/docs")
            print(f"📊 Health Check: http://{host}:{port}/health")
            print(f"📈 Metrics: http://{host}:{port}/metrics")

        # Start the server
        uvicorn.run(
            "api_server:app",
            host=host,
            port=port,
            reload=dev,
            log_level="debug" if dev else "info",
        )

    except ImportError:
        print("❌ API server dependencies not available.")
        print("   Install with: uv sync")
        raise click.Abort() from None
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        raise click.Abort() from e


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option("--steps", type=int, default=5, help="Number of forecast steps ahead")
@click.option(
    "--timeframe",
    type=click.Choice(["1D", "1H", "15m"]),
    default="1D",
    help="Timeframe",
)
@click.option("--n-states", type=int, default=6, help="Number of HMM states")
@click.pass_context
@handle_exceptions
def regime_forecast(
    ctx: click.Context, symbol: str, steps: int, timeframe: str, n_states: int
) -> None:
    """Forecast future regime probabilities using the HMM transition matrix.

    Uses the learned transition matrix to project the current regime state
    distribution forward by N steps, showing how regime probabilities evolve.
    Also displays regime stability metrics (expected durations, stationary distribution).

    Args:
        ctx: Click context containing global options
        symbol: Trading symbol to analyze
        steps: Number of forecast steps ahead
        timeframe: Time interval for analysis
        n_states: Number of HMM states
    """
    provider_name = ctx.obj["provider"]
    api_key = ctx.obj["api_key"]

    print(f"\nFetching {timeframe} data for {symbol}...")

    # Map timeframe to period/interval for provider
    tf_map = {"1D": ("2y", "1d"), "1H": ("6mo", "1h"), "15m": ("2mo", "15m")}
    period, interval = tf_map[timeframe]

    provider = MarketDataProvider.create_provider(provider_name, api_key=api_key)
    df = provider.fetch(symbol, period, interval)

    print(f"Training HMM with {n_states} states on {len(df)} bars...")
    hmm = TrueHMMDetector(n_states=n_states, n_iter=100)
    hmm.fit(df)

    # Current regime
    regime, state, confidence = hmm.predict_regime(df)
    print(f"\nCURRENT REGIME: {regime.value} (state {state}, confidence {confidence:.1%})")

    # Forecast
    print(f"\nREGIME FORECAST ({steps}-step ahead):")
    print("=" * 80)

    forecasts = hmm.forecast_regime_sequence(df, n_steps=steps)
    state_regime_map = hmm.get_state_regime_map()

    # Collect all regimes that appear
    all_regimes = sorted(
        {r for f in forecasts for r in f["regime_probabilities"]},
        key=lambda r: r.value,
    )

    # Header
    header = f"{'Step':>5}"
    for r in all_regimes:
        header += f"  {r.value:>16}"
    header += f"  {'Most Likely':>18}"
    print(header)
    print("-" * 80)

    for f in forecasts:
        row = f"{f['step']:>5}"
        for r in all_regimes:
            prob = f["regime_probabilities"].get(r, 0.0)
            row += f"  {prob:>16.1%}"
        row += f"  {f['most_likely_regime'].value:>18}"
        print(row)

    # Stability metrics
    print("\nREGIME STABILITY METRICS:")
    print("=" * 80)

    stability = hmm.get_regime_stability()

    print(f"\n{'State':>6} {'Regime':<20} {'Self-Trans':>10} {'Exp Duration':>13}")
    print("-" * 55)
    for i in range(hmm.n_states):
        r = state_regime_map[i]
        st = stability["self_transition_probs"][i]
        dur = stability["expected_durations"][i]
        max_display_duration = 1000
        dur_str = f"{dur:.1f}" if dur < max_display_duration else "inf"
        print(f"{i:>6} {r.value:<20} {st:>10.1%} {dur_str:>13}")

    print("\nSTATIONARY DISTRIBUTION (long-run regime probabilities):")
    print("-" * 55)
    for r in sorted(stability["stationary_regimes"], key=lambda r: r.value):
        prob = stability["stationary_regimes"][r]
        bar = "#" * int(prob * 40)
        print(f"  {r.value:<20} {prob:>7.1%}  {bar}")


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--method",
    type=click.Choice(["sharpe_weighted", "win_rate", "profit_factor", "kelly"]),
    default="sharpe_weighted",
    help="Calibration scoring method",
)
@click.option("--output", type=str, default=None, help="Save calibrated params to JSON file")
@click.option("--n-states", type=int, default=4, help="Number of HMM states")
@click.pass_context
@handle_exceptions
def calibrate_multipliers(
    ctx: click.Context,
    symbol: str,
    method: str,
    output: str | None,
    n_states: int,
) -> None:
    """Empirically calibrate regime multipliers from historical backtest data.

    Runs a walk-forward backtest with uniform multipliers, then analyzes
    per-regime trade performance to derive optimal position size multipliers.

    Args:
        ctx: Click context containing global options
        symbol: Trading symbol to analyze
        method: Calibration scoring method
        output: Optional path to save results as JSON
        n_states: Number of HMM states
    """
    import json  # noqa: PLC0415

    provider_name = ctx.obj["provider"]
    api_key = ctx.obj["api_key"]

    print(f"\nFetching daily data for {symbol}...")
    provider = MarketDataProvider.create_provider(provider_name, api_key=api_key)
    df = provider.fetch(symbol, "2y", "1d")
    print(f"Loaded {len(df)} bars.")

    calibrator = RegimeMultiplierCalibrator(
        df=df,
        n_hmm_states=n_states,
        hmm_n_iter=50,
        retrain_frequency=20,
        min_train_days=252,
        test_days=63,
    )

    result = calibrator.calibrate_with_details(method=method, verbose=True)

    print(f"\nBaseline Sharpe: {result.baseline_sharpe:.2f}")
    print(f"Total trades analyzed: {result.total_trades}")

    # Save to JSON if requested
    if output:
        output_data = {
            "symbol": symbol,
            "method": method,
            "total_trades": result.total_trades,
            "baseline_sharpe": result.baseline_sharpe,
            "multipliers": {r.value: v for r, v in result.multipliers.items()},
            "trades_per_regime": {r.value: v for r, v in result.trades_per_regime.items()},
            "raw_scores": {r.value: v for r, v in result.raw_scores.items()},
            "regime_stats": {
                r.value: {
                    "n_trades": rs.n_trades,
                    "win_rate": rs.win_rate,
                    "avg_pnl": rs.avg_pnl,
                    "sharpe": rs.sharpe,
                    "profit_factor": rs.profit_factor,
                    "kelly_fraction": rs.kelly_fraction,
                }
                for r, rs in result.regime_stats.items()
            },
        }
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output}")


if __name__ == "__main__":
    cli()
