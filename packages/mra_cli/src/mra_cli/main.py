#!/usr/bin/env python3
"""
Market Regime Analysis - command line interface (Click).

Hidden Markov Model regime detection, regime forecasting, and position sizing
from the terminal. Data comes from a pluggable provider (``--provider``); the
default is Yahoo Finance (no key needed) and ``--provider mock`` works offline.
"""

import concurrent.futures
import contextlib
import functools
import io
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, cast

import click

from mra_lib import (
    MarketRegime,
    MarketRegimeAnalyzer,
    PortfolioHMMAnalyzer,
    SimonsRiskCalculator,
)
from mra_lib.backtesting import RegimeMultiplierCalibrator
from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.timeframes import (
    BARS_PER_DAY,
    DEFAULT_PERIODS,
    TIMEFRAME_INTERVALS,
    TIMEFRAMES,
)
from mra_lib.data_providers import (
    AuthError,
    InvalidSymbolError,
    MarketDataProvider,
    ProviderError,
    RateLimitError,
    required_env_vars,
    resolve_api_key,
)
from mra_lib.data_providers.credentials import PROVIDER_ENV_PAIRS
from mra_lib.indicators.true_hmm_detector import TrueHMMDetector

F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_PROVIDER_ENV = "DEFAULT_PROVIDER"
FALLBACK_PROVIDER = "yfinance"

# Provider names come from the registry, so newly registered providers show up here
PROVIDER_CHOICES = sorted(MarketDataProvider.get_available_providers())
TIMEFRAME_CHOICE = click.Choice(list(TIMEFRAMES))

# Upper bound for the retry delay after repeated monitoring failures
_MAX_BACKOFF_SECONDS = 3600.0


def validate_api_key(provider: str, api_key: str | None) -> str:
    """Validate and retrieve API key for providers that require it.

    Args:
        provider: Data provider name
        api_key: Optional API key from command line

    Returns:
        str: Valid API key ("" for providers that need none)

    Raises:
        click.ClickException: If required API key is missing
    """
    resolved = resolve_api_key(provider, api_key)
    if resolved is not None:
        return resolved

    provider_name = provider.replace("_", " ").title()
    # Key-pair providers need every variable; otherwise any one of them will do
    joiner = " and " if provider in PROVIDER_ENV_PAIRS else " or "
    env_vars = joiner.join(required_env_vars(provider))
    raise click.ClickException(
        f"{provider_name} API key is required when using {provider} provider. "
        f"Set {env_vars} or use --api-key option."
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


def _debug_enabled() -> bool:
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    obj = ctx.find_root().obj
    return bool(obj.get("debug")) if isinstance(obj, dict) else False


def _provider_cause(error: BaseException) -> BaseException:
    """Return the first ``ProviderError`` in the cause/context chain, else ``error`` itself."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, ProviderError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return error


def handle_exceptions(func: F) -> F:
    """Report errors consistently and exit non-zero; ``--debug`` re-raises with traceback."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (click.ClickException, click.Abort, click.exceptions.Exit):
            raise
        except Exception as e:
            if _debug_enabled():
                raise
            # The analyzer re-wraps provider errors as ValueError; label by the root cause
            cause = _provider_cause(e)
            if isinstance(cause, AuthError):
                label = "🔑 Authentication error"
            elif isinstance(cause, RateLimitError):
                label = "⏳ Rate limited"
            elif isinstance(cause, InvalidSymbolError):
                label = "❓ Unknown symbol / no data"
            elif isinstance(e, ValueError):
                label = "❌ Invalid input"
            elif isinstance(e, ConnectionError):
                label = "🌐 Network error"
            elif isinstance(e, FileNotFoundError):
                label = "📁 File not found"
            elif isinstance(e, PermissionError):
                label = "🔒 Permission error"
            else:
                label = "💥 Unexpected error"
            click.echo(f"{label}: {e}", err=True)
            raise click.Abort() from e

    return cast(F, wrapper)


def provider_options(func: F) -> F:
    """Add ``--provider`` / ``--api-key`` to a subcommand (overrides the group options)."""
    func = click.option(
        "--api-key",
        type=str,
        default=None,
        help="API key (prefer environment variables; CLI args end up in shell history)",
    )(func)
    func = click.option(
        "--provider",
        type=click.Choice(PROVIDER_CHOICES),
        default=None,
        help=f"Data provider (default: ${DEFAULT_PROVIDER_ENV} or {FALLBACK_PROVIDER})",
    )(func)
    return func


def resolve_provider(
    ctx: click.Context, provider: str | None, api_key: str | None
) -> tuple[str, str | None]:
    """Pick the provider (subcommand > group > $DEFAULT_PROVIDER > yfinance) and its key.

    The key is resolved lazily here, so commands that never fetch data work without one.

    Returns:
        ``(provider_name, api_key)``; the key is ``None`` for keyless providers
    """
    obj = ctx.find_root().obj or {}
    name = (
        provider
        or obj.get("provider")
        or os.getenv(DEFAULT_PROVIDER_ENV, "").strip().lower()
        or FALLBACK_PROVIDER
    )
    if name not in MarketDataProvider.get_available_providers():
        raise click.ClickException(
            f"Unknown provider '{name}' (from ${DEFAULT_PROVIDER_ENV}?). "
            f"Available: {', '.join(PROVIDER_CHOICES)}"
        )
    # A group-level key only applies to the group-level provider choice
    group_key = obj.get("api_key") if not provider or provider == obj.get("provider") else None
    key = validate_api_key(name, api_key or group_key)
    return name, key or None


def print_regime_report(
    symbol: str, timeframe: str, analysis: RegimeAnalysis, current_price: float | None
) -> None:
    """Print a formatted report for an already-computed analysis."""
    click.echo("\n" + "=" * 80)
    click.echo(f"HMM MARKET REGIME ANALYSIS - {symbol} ({timeframe})")
    click.echo("=" * 80)
    if current_price is not None:
        click.echo(f"Current Price: ${current_price:.2f}")
    click.echo(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    click.echo("\n📊 REGIME CLASSIFICATION:")
    click.echo(f"   Current Regime: {analysis.current_regime.value}")
    click.echo(f"   HMM State: {analysis.hmm_state}")
    click.echo(f"   Confidence: {analysis.regime_confidence:.1%}")
    click.echo(f"   Persistence: {analysis.regime_persistence:.1%}")
    click.echo(f"   Transition Prob: {analysis.transition_probability:.1%}")

    click.echo("\n📈 TRADING RECOMMENDATION:")
    click.echo(f"   Strategy: {analysis.recommended_strategy.value}")
    click.echo(f"   Position Multiplier: {analysis.position_sizing_multiplier:.2f}x")
    click.echo(f"   Risk Level: {analysis.risk_level}")

    if analysis.arbitrage_opportunities:
        click.echo("\n💰 STATISTICAL ARBITRAGE:")
        for opp in analysis.arbitrage_opportunities:
            click.echo(f"   • {opp}")

    if analysis.statistical_signals:
        click.echo("\n📡 STATISTICAL SIGNALS:")
        for signal in analysis.statistical_signals:
            click.echo(f"   • {signal}")

    if analysis.key_levels:
        click.echo("\n🎯 KEY LEVELS:")
        for level_name, level_value in analysis.key_levels.items():
            click.echo(f"   {level_name.upper()}: ${level_value:.2f}")

    click.echo("=" * 80)


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats (inf/nan, e.g. profit factor) with ``None``."""
    import math  # noqa: PLC0415

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and not isinstance(value, str):  # numpy scalars
        return _json_safe(value.item())
    return value


def _last_close(analyzer: MarketRegimeAnalyzer, timeframe: str) -> float | None:
    df = analyzer.data.get(timeframe)
    if df is None or df.empty:
        return None
    return float(df["Close"].iloc[-1])


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


def _analyze_single_timeframe(
    symbol: str, timeframe: str, provider: str, api_key: str | None
) -> tuple[MarketRegimeAnalyzer, RegimeAnalysis]:
    """Load one timeframe, train its model, and analyze it."""
    analyzer = MarketRegimeAnalyzer(
        symbol,
        periods={timeframe: DEFAULT_PERIODS[timeframe]},
        provider_flag=provider,
        api_key=api_key,
    )
    return analyzer, analyzer.analyze_current_regime(timeframe)


@click.group()
@click.option("--debug/--no-debug", default=False, help="Verbose logs; show tracebacks on errors")
@click.option(
    "--provider",
    type=click.Choice(PROVIDER_CHOICES),
    default=None,
    help=f"Data provider for all commands (default: ${DEFAULT_PROVIDER_ENV} or yfinance)",
)
@click.option(
    "--api-key",
    type=str,
    help="API key (SECURITY WARNING: prefer environment variables over CLI arguments)",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool, provider: str | None, api_key: str | None) -> None:
    """
    Market Regime Analysis CLI.

    Detects market regimes with Hidden Markov Models and derives regime-aware
    trading signals and position sizing. Research tool only; not financial advice.

    \b
    --provider / --api-key can be given before or after the subcommand:
      mra --provider mock current-analysis --symbol SPY
      mra current-analysis --provider mock --symbol SPY

    \b
    🌐 Web API: run 'uv run mra-api --dev' (docs at http://localhost:8000/docs)
    🐍 Python client example: examples/api_client.py
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["provider"] = provider
    ctx.obj["api_key"] = api_key

    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.WARNING)
    logging.getLogger("mra_lib").setLevel(logging.DEBUG if debug else logging.INFO)


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option("--timeframe", type=TIMEFRAME_CHOICE, default="1D", help="Timeframe")
@provider_options
@click.pass_context
@handle_exceptions
def detailed_analysis(
    ctx: click.Context,
    symbol: str,
    timeframe: str,
    provider: str | None,
    api_key: str | None,
) -> None:
    """Run detailed HMM analysis for a single timeframe."""
    provider_name, key = resolve_provider(ctx, provider, api_key)
    click.echo(f"\nInitializing detailed analysis for {symbol} ({timeframe})...")

    analyzer, analysis = _analyze_single_timeframe(symbol, timeframe, provider_name, key)
    print_regime_report(symbol, timeframe, analysis, _last_close(analyzer, timeframe))

    click.echo("\n📋 DETAILED METRICS:")
    click.echo(f"   HMM State: {analysis.hmm_state}")
    click.echo(f"   Regime: {analysis.current_regime.value}")
    click.echo(f"   Confidence: {analysis.regime_confidence:.3f}")
    click.echo(f"   Persistence: {analysis.regime_persistence:.3f}")
    click.echo(f"   Transition Prob: {analysis.transition_probability:.3f}")
    click.echo("\n⚠️  RISK ASSESSMENT:")
    click.echo(f"   Risk Level: {analysis.risk_level}")
    click.echo(f"   Position Multiplier: {analysis.position_sizing_multiplier:.3f}")
    click.echo(f"   Strategy: {analysis.recommended_strategy.value}")


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@provider_options
@click.pass_context
@handle_exceptions
def current_analysis(
    ctx: click.Context, symbol: str, provider: str | None, api_key: str | None
) -> None:
    """Run current HMM regime analysis for all timeframes.

    Each timeframe is loaded and analyzed independently, so one failing timeframe
    does not hide the others. Exits non-zero only if every timeframe fails.
    """
    provider_name, key = resolve_provider(ctx, provider, api_key)
    click.echo(f"\nAnalyzing {symbol} on {', '.join(TIMEFRAMES)} with {provider_name}...")

    results: dict[str, tuple[MarketRegimeAnalyzer, RegimeAnalysis] | Exception] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(TIMEFRAMES)) as executor:
        futures = {
            tf: executor.submit(_analyze_single_timeframe, symbol, tf, provider_name, key)
            for tf in TIMEFRAMES
        }
        for tf, future in futures.items():
            try:
                results[tf] = future.result()
            except Exception as e:
                if _debug_enabled():
                    raise
                results[tf] = e

    failed = 0
    for tf in TIMEFRAMES:
        result = results[tf]
        if isinstance(result, Exception):
            failed += 1
            click.echo(f"\n❌ Error analyzing {tf}: {result}", err=True)
        else:
            analyzer, analysis = result
            print_regime_report(symbol, tf, analysis, _last_close(analyzer, tf))

    if failed == len(TIMEFRAMES):
        raise click.ClickException(f"Analysis failed for every timeframe of {symbol}")


# Messages the library prints (instead of raising) when plotting fails
_CHART_FAILURE_MARKERS = ("Error generating chart", "Insufficient data for plotting")
_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}


class _Tee(io.TextIOBase):
    """Text stream that forwards writes to ``target`` and keeps a copy."""

    def __init__(self, target: Any) -> None:
        self._target = target
        self._parts: list[str] = []

    def write(self, text: str) -> int:
        self._parts.append(text)
        return int(self._target.write(text))

    def flush(self) -> None:
        self._target.flush()

    @property
    def captured(self) -> str:
        return "".join(self._parts)


def _new_figure(before: set[int]) -> Any:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    new = [n for n in plt.get_fignums() if n not in before]
    return plt.figure(new[-1]) if new else None


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option("--timeframe", type=TIMEFRAME_CHOICE, default="1D", help="Timeframe")
@click.option(
    "--days",
    type=click.IntRange(min=1),
    default=60,
    help="Trading days to plot (converted to bars for intraday timeframes)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Save the chart to this file (PNG/SVG/PDF by extension) instead of showing it",
)
@provider_options
@click.pass_context
@handle_exceptions
def generate_charts(  # noqa: PLR0913, PLR0917
    ctx: click.Context,
    symbol: str,
    timeframe: str,
    days: int,
    output: Path | None,
    provider: str | None,
    api_key: str | None,
) -> None:
    """Generate HMM charts for a given symbol and timeframe."""
    import matplotlib  # noqa: PLC0415

    if output is not None:
        matplotlib.use("Agg", force=True)  # Headless rendering
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    provider_name, key = resolve_provider(ctx, provider, api_key)
    click.echo(f"Initializing analyzer for {symbol}...")
    analyzer = MarketRegimeAnalyzer(
        symbol,
        periods={timeframe: DEFAULT_PERIODS[timeframe]},
        provider_flag=provider_name,
        api_key=key,
    )

    bars = days * BARS_PER_DAY[timeframe]
    click.echo(f"Generating charts for {timeframe} ({days} days, {bars} bars)...")

    # Defer any plt.show() inside the library so we can verify a chart was produced, and
    # watch its output: the current library prints errors instead of raising them
    before = set(plt.get_fignums())
    real_show = plt.show
    plt.show = lambda *a, **k: None  # type: ignore[assignment]
    tee = _Tee(sys.stdout)
    try:
        with contextlib.redirect_stdout(tee):
            result = analyzer.plot_regime_analysis(timeframe, bars)  # type: ignore[func-returns-value]
    finally:
        plt.show = real_show  # type: ignore[assignment]

    if any(marker in tee.captured for marker in _CHART_FAILURE_MARKERS):
        for num in set(plt.get_fignums()) - before:
            plt.close(num)
        raise click.ClickException(f"Chart generation failed for {symbol} ({timeframe})")

    # Nobody would see a figure on a non-interactive backend: save it instead
    if output is None and matplotlib.get_backend().lower() in _NON_INTERACTIVE_BACKENDS:
        output = Path(f"{symbol}_{timeframe}_regimes.png")

    if isinstance(result, bytes | bytearray):
        target = output or Path(f"{symbol}_{timeframe}_regimes.png")
        target.write_bytes(bytes(result))
        click.echo(f"✓ Chart saved to {target}")
        return

    fig = result if isinstance(result, Figure) else _new_figure(before)
    if fig is None:
        raise click.ClickException(f"Chart generation failed for {symbol} ({timeframe})")

    if output is not None:
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        click.echo(f"✓ Chart saved to {output}")
    else:
        plt.show()


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option("--filename", type=str, default=None, help="Filename for CSV export")
@provider_options
@click.pass_context
@handle_exceptions
def export_csv(
    ctx: click.Context,
    symbol: str,
    filename: str | None,
    provider: str | None,
    api_key: str | None,
) -> None:
    """Export HMM analysis to CSV for a given symbol."""
    import pandas as pd  # noqa: PLC0415

    provider_name, key = resolve_provider(ctx, provider, api_key)
    if filename is None:
        filename = f"{symbol}_hmm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    target = Path(filename)

    click.echo(f"Initializing analyzer for {symbol}...")
    analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider_name, api_key=key)

    click.echo("Exporting analysis data...")
    started = time.time()
    result = analyzer.export_analysis_to_csv(filename)  # type: ignore[func-returns-value]

    # Newer library versions may return the data (or a path) instead of writing it
    if isinstance(result, pd.DataFrame):
        if result.empty:
            raise click.ClickException("No analysis data to export")
        if not target.exists() or target.stat().st_mtime < started - 1:
            result.to_csv(target, index=not isinstance(result.index, pd.RangeIndex))
    elif isinstance(result, str | Path):
        target = Path(result)

    if not target.exists() or target.stat().st_mtime < started - 1 or target.stat().st_size == 0:
        raise click.ClickException(f"Export failed: {target} was not written")
    click.echo(f"✓ Analysis exported to {target}")


@cli.command()
def list_providers() -> None:
    """List all available data providers and their capabilities."""
    providers = MarketDataProvider.get_available_providers()

    click.echo("\n📡 AVAILABLE DATA PROVIDERS:")
    click.echo("=" * 50)

    for name, info in sorted(providers.items()):
        click.echo(f"\n🔹 {name.upper()}")
        click.echo(f"   Description: {info['description']}")
        click.echo(f"   Requires API Key: {'Yes' if info['requires_api_key'] else 'No'}")
        rate = info["rate_limit_per_minute"]
        click.echo(f"   Rate Limit: {f'{rate} req/min' if rate else 'none'}")
        click.echo(f"   Supported Intervals: {', '.join(sorted(info['supported_intervals']))}")
        click.echo(f"   Supported Periods: {', '.join(sorted(info['supported_periods']))}")

    click.echo("\n💡 USAGE EXAMPLES:")
    click.echo("   mra current-analysis --symbol SPY                     # yfinance, no key")
    click.echo("   mra current-analysis --provider mock --symbol SPY     # offline data")
    click.echo("   mra current-analysis --provider polygon --symbol SPY  # key from env")
    click.echo(f"   export {DEFAULT_PROVIDER_ENV}=tiingo                       # change default")
    click.echo("   export ALPHA_VANTAGE_API_KEY=your_key")
    click.echo("   export POLYGON_API_KEY=your_key")
    click.echo("   export APCA_API_KEY_ID=your_key_id APCA_API_SECRET_KEY=your_secret")
    click.echo("   export TIINGO_API_KEY=your_key")


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
    """Calculate position sizing based on regime, confidence, persistence, and correlation."""
    regime_enum = next(r for r in MarketRegime if r.value == regime)
    size = SimonsRiskCalculator.calculate_regime_adjusted_size(
        base_size, regime_enum, confidence, persistence
    )
    correlation_adjusted = SimonsRiskCalculator.calculate_correlation_adjusted_size(
        size, correlation
    )
    click.echo("\n📊 POSITION SIZING RESULTS:")
    click.echo(f"   Base Size: {base_size:.1%}")
    click.echo(f"   Regime: {regime}")
    click.echo(f"   Regime Adjusted: {size:.1%}")
    click.echo(f"   Correlation Adjusted: {correlation_adjusted:.1%}")
    click.echo(f"   Final Recommendation: {correlation_adjusted:.1%}")


@cli.command()
@click.option(
    "--symbols",
    type=str,
    default="SPY,QQQ,IWM",
    help="Comma-separated list of symbols (e.g., SPY,QQQ,IWM)",
)
@click.option("--timeframe", type=TIMEFRAME_CHOICE, default="1D", help="Timeframe")
@provider_options
@click.pass_context
@handle_exceptions
def multi_symbol_analysis(
    ctx: click.Context,
    symbols: str,
    timeframe: str,
    provider: str | None,
    api_key: str | None,
) -> None:
    """Run multi-symbol HMM analysis (portfolio)."""
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise click.BadParameter("At least one symbol must be provided", param_hint="--symbols")

    provider_name, key = resolve_provider(ctx, provider, api_key)
    portfolio = PortfolioHMMAnalyzer(
        symbol_list,
        periods={timeframe: DEFAULT_PERIODS[timeframe]},
        provider_flag=provider_name,
        api_key=key,
    )
    if not portfolio.analyzers:
        raise click.ClickException("No symbol could be loaded")
    portfolio.print_portfolio_summary(timeframe)


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--interval",
    type=click.IntRange(min=1),
    default=300,
    help="Refresh interval in seconds (default: 300)",
)
@click.option("--once", is_flag=True, help="Run a single iteration and exit")
@click.option(
    "--max-iterations",
    type=click.IntRange(min=1),
    default=None,
    help="Stop after this many iterations (default: run until Ctrl+C / SIGTERM)",
)
@provider_options
@click.pass_context
@handle_exceptions
def continuous_monitoring(  # noqa: PLR0913, PLR0917
    ctx: click.Context,
    symbol: str,
    interval: int,
    once: bool,
    max_iterations: int | None,
    provider: str | None,
    api_key: str | None,
) -> None:
    """Continuously refresh and report regimes for a symbol.

    Transient failures are retried with exponential backoff instead of exiting.
    Exits non-zero if no iteration succeeded.
    """
    provider_name, key = resolve_provider(ctx, provider, api_key)
    click.echo(f"Starting continuous monitoring for {symbol} (Ctrl+C to stop)...")

    limit = 1 if once else max_iterations

    # The constructor loads data, so retry it with the same backoff as the monitoring loop;
    # each failed start counts as an iteration
    attempts = 0
    while True:
        attempts += 1
        try:
            analyzer = MarketRegimeAnalyzer(symbol, provider_flag=provider_name, api_key=key)
            break
        except Exception as e:
            if _debug_enabled() or (limit is not None and attempts >= limit):
                raise
            delay = min(max(_MAX_BACKOFF_SECONDS, interval), interval * 2 ** (attempts - 1))
            click.echo(f"⚠️  Startup failed ({e}); retrying in {delay:.0f}s", err=True)
            time.sleep(delay)

    def report(timeframe: str, analysis: RegimeAnalysis) -> None:
        print_regime_report(symbol, timeframe, analysis, _last_close(analyzer, timeframe))

    successes = analyzer.run_continuous_monitoring(
        interval,
        max_iterations=None if limit is None else limit - attempts + 1,
        max_backoff=_MAX_BACKOFF_SECONDS,
        on_update=report,
    )
    if successes == 0:
        raise click.ClickException("Monitoring finished without a successful iteration")


@cli.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="API server host (use 0.0.0.0 to listen on all interfaces)",
)
@click.option("--port", type=click.IntRange(1, 65535), default=8000, help="API server port")
@click.option("--dev/--no-dev", default=False, help="Development mode with auto-reload")
@handle_exceptions
def start_api(host: str, port: int, dev: bool) -> None:
    """Start the REST API server for web access.

    \b
    Examples:
        uv run mra start-api --dev              # Development mode (localhost)
        uv run mra start-api --host 0.0.0.0     # Listen on all interfaces
    """
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        raise click.ClickException(
            "API server dependencies not available. Install with: uv sync"
        ) from None

    click.echo("🚀 Starting Market Regime Analysis API Server")
    click.echo(f"🌐 Host: {host}")
    click.echo(f"🔌 Port: {port}")
    click.echo(f"🔄 Development mode: {dev}")

    if dev:
        click.echo(f"📖 API Documentation: http://{host}:{port}/docs")
        click.echo(f"📊 Health Check: http://{host}:{port}/health")
        click.echo(f"📈 Metrics: http://{host}:{port}/metrics")

    uvicorn.run(
        "mra_web.app:app",
        host=host,
        port=port,
        reload=dev,
        log_level="debug" if dev else "info",
    )


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--steps", type=click.IntRange(1, 1000), default=5, help="Number of forecast steps ahead"
)
@click.option("--timeframe", type=TIMEFRAME_CHOICE, default="1D", help="Timeframe")
@click.option("--n-states", type=click.IntRange(2, 12), default=6, help="Number of HMM states")
@provider_options
@click.pass_context
@handle_exceptions
def regime_forecast(  # noqa: PLR0913, PLR0917
    ctx: click.Context,
    symbol: str,
    steps: int,
    timeframe: str,
    n_states: int,
    provider: str | None,
    api_key: str | None,
) -> None:
    """Forecast future regime probabilities using the HMM transition matrix.

    Uses the learned transition matrix to project the current regime state
    distribution forward by N steps, showing how regime probabilities evolve.
    Also displays regime stability metrics (expected durations, stationary distribution).
    """
    provider_name, key = resolve_provider(ctx, provider, api_key)
    click.echo(f"\nFetching {timeframe} data for {symbol}...")

    period, interval = DEFAULT_PERIODS[timeframe], TIMEFRAME_INTERVALS[timeframe]
    data_provider = MarketDataProvider.create_provider(provider_name, api_key=key)
    df = data_provider.fetch(symbol, period, interval)

    click.echo(f"Training HMM with {n_states} states on {len(df)} bars...")
    hmm = TrueHMMDetector(n_states=n_states, n_iter=100)
    hmm.fit(df)

    # Current regime
    regime, state, confidence = hmm.predict_regime(df)
    click.echo(f"\nCURRENT REGIME: {regime.value} (state {state}, confidence {confidence:.1%})")

    # Forecast
    click.echo(f"\nREGIME FORECAST ({steps}-step ahead):")
    click.echo("=" * 80)

    forecasts = hmm.forecast_regime_sequence(df, n_steps=steps)
    state_regime_map = hmm.get_state_regime_map()

    # Collect all regimes that appear
    all_regimes = sorted(
        {r for f in forecasts for r in f["regime_probabilities"]},
        key=lambda r: r.value,
    )

    header = f"{'Step':>5}"
    for r in all_regimes:
        header += f"  {r.value:>16}"
    header += f"  {'Most Likely':>18}"
    click.echo(header)
    click.echo("-" * 80)

    for f in forecasts:
        row = f"{f['step']:>5}"
        for r in all_regimes:
            prob = f["regime_probabilities"].get(r, 0.0)
            row += f"  {prob:>16.1%}"
        row += f"  {f['most_likely_regime'].value:>18}"
        click.echo(row)

    # Stability metrics
    click.echo("\nREGIME STABILITY METRICS:")
    click.echo("=" * 80)

    stability = hmm.get_regime_stability()

    click.echo(f"\n{'State':>6} {'Regime':<20} {'Self-Trans':>10} {'Exp Duration':>13}")
    click.echo("-" * 55)
    max_display_duration = 1000
    for i in range(hmm.n_states):
        r = state_regime_map[i]
        st = stability["self_transition_probs"][i]
        dur = stability["expected_durations"][i]
        dur_str = f"{dur:.1f}" if dur < max_display_duration else "inf"
        click.echo(f"{i:>6} {r.value:<20} {st:>10.1%} {dur_str:>13}")

    click.echo("\nSTATIONARY DISTRIBUTION (long-run regime probabilities):")
    click.echo("-" * 55)
    for r in sorted(stability["stationary_regimes"], key=lambda r: r.value):
        prob = stability["stationary_regimes"][r]
        bar = "#" * int(prob * 40)
        click.echo(f"  {r.value:<20} {prob:>7.1%}  {bar}")


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--method",
    type=click.Choice(["sharpe_weighted", "win_rate", "profit_factor", "kelly"]),
    default="sharpe_weighted",
    help="Calibration scoring method",
)
@click.option("--output", type=str, default=None, help="Save calibrated params to JSON file")
@click.option("--n-states", type=click.IntRange(2, 12), default=4, help="Number of HMM states")
@provider_options
@click.pass_context
@handle_exceptions
def calibrate_multipliers(  # noqa: PLR0913, PLR0917
    ctx: click.Context,
    symbol: str,
    method: str,
    output: str | None,
    n_states: int,
    provider: str | None,
    api_key: str | None,
) -> None:
    """Empirically calibrate regime multipliers from historical backtest data.

    Runs a walk-forward backtest with uniform multipliers, then analyzes
    per-regime trade performance to derive optimal position size multipliers.
    """
    import json  # noqa: PLC0415

    provider_name, key = resolve_provider(ctx, provider, api_key)

    click.echo(f"\nFetching daily data for {symbol}...")
    data_provider = MarketDataProvider.create_provider(provider_name, api_key=key)
    df = data_provider.fetch(symbol, DEFAULT_PERIODS["1D"], TIMEFRAME_INTERVALS["1D"])
    click.echo(f"Loaded {len(df)} bars.")

    calibrator = RegimeMultiplierCalibrator(
        df=df,
        n_hmm_states=n_states,
        hmm_n_iter=50,
        retrain_frequency=20,
        min_train_days=252,
        test_days=63,
    )

    result = calibrator.calibrate_with_details(method=method, verbose=True)

    click.echo(f"\nBaseline Sharpe: {result.baseline_sharpe:.2f}")
    click.echo(f"Total trades analyzed: {result.total_trades}")

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
            json.dump(_json_safe(output_data), f, indent=2, allow_nan=False)
        click.echo(f"\nResults saved to {output}")


if __name__ == "__main__":
    cli()
