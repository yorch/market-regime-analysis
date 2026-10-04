#!/usr/bin/env python3
"""
Market Regime Analysis - command line interface (Click).

Hidden Markov Model regime detection, regime forecasting, and position sizing
from the terminal. Data comes from a pluggable provider (``--provider``); the
default is Yahoo Finance (no key needed) and ``--provider mock`` works offline.
"""

import concurrent.futures
import functools
import logging
import os
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
from mra_lib.config.env_file import EnvFileError, load_env_file
from mra_lib.config.timeframes import (
    BARS_PER_DAY,
    DEFAULT_PERIODS,
    MIN_CONFIRMATION_TIMEFRAMES,
    TIMEFRAME_INTERVALS,
    TIMEFRAMES,
)
from mra_lib.data_providers import (
    AuthError,
    InvalidSymbolError,
    MarketDataProvider,
    RateLimitError,
    required_env_vars,
    resolve_api_key,
)
from mra_lib.data_providers.credentials import PROVIDER_ENV_PAIRS
from mra_lib.errors import InsufficientDataError, StorageError
from mra_lib.indicators.true_hmm_detector import TrueHMMDetector
from mra_lib.scanner import (
    DEFAULT_COOLDOWN_BARS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_WATCH_TIMEFRAMES,
    AlertPolicy,
    LogNotifier,
    Notifier,
    Scanner,
    ScanReport,
    format_scan_report,
    notifiers_from_env,
)
from mra_lib.scanner.notifiers import describe_notifier
from mra_lib.signals.confirmation import confirm_timeframes, format_confirmation_report
from mra_lib.storage import MAX_HISTORY_LIMIT, RegimeRecord, RegimeStore, default_store

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


class _CliLogFormatter(logging.Formatter):
    """``INFO`` records print as the bare message (progress lines); others get a prefix."""

    def __init__(self) -> None:
        super().__init__("%(levelname)s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return record.getMessage()
        return super().format(record)


class _ClickEchoHandler(logging.Handler):
    """Write log records to stderr through ``click.echo`` (resolved at emit time).

    Resolving the stream per record keeps output correct when the CLI is invoked
    repeatedly in one process (e.g. ``CliRunner``), unlike a ``StreamHandler``
    bound to the stderr object that existed at configuration time.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            click.echo(self.format(record), err=True)
        except Exception:  # noqa: BLE001 - logging must never raise; report via handleError
            self.handleError(record)


def configure_logging(debug: bool) -> None:
    """Route log records to stderr; show ``mra_lib`` progress (INFO) as plain lines.

    The library never prints, it logs. Other libraries only surface warnings,
    ``mra_lib`` logs at INFO (DEBUG with ``--debug``). Idempotent.
    """
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h, _ClickEchoHandler)]:
        root.removeHandler(handler)
    handler = _ClickEchoHandler()
    handler.setFormatter(_CliLogFormatter())
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.WARNING:
        root.setLevel(logging.WARNING)
    logging.getLogger("mra_lib").setLevel(logging.DEBUG if debug else logging.INFO)


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
            # The library raises provider errors unchanged, so label by type
            if isinstance(e, AuthError):
                label = "🔑 Authentication error"
            elif isinstance(e, RateLimitError):
                label = "⏳ Rate limited"
            elif isinstance(e, InvalidSymbolError):
                label = "❓ Unknown symbol / no data"
            elif isinstance(e, StorageError) and not isinstance(e, ValueError):
                label = "🗄️  Storage error"
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
    except Exception as e:  # noqa: BLE001 - returned to the caller as the result
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
    try:  # before any env read ($DEFAULT_PROVIDER, provider keys, MRA_DB_PATH)
        load_env_file()
    except EnvFileError as e:
        raise click.ClickException(str(e)) from e
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["provider"] = provider
    ctx.obj["api_key"] = api_key

    configure_logging(debug)


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


def _record_analyses(
    store: RegimeStore, analyses: list[tuple[str, MarketRegimeAnalyzer, RegimeAnalysis]]
) -> int:
    """Save one regime record per analyzed timeframe; report failures without raising.

    Returns:
        The number of records saved.
    """
    saved = 0
    for tf, analyzer, analysis in analyses:
        try:
            store.save(RegimeRecord.from_analysis(analysis, analyzer, tf))
            saved += 1
        except Exception as e:
            if _debug_enabled():
                raise
            logging.getLogger(__name__).warning("Recording %s failed", tf, exc_info=True)
            click.echo(f"⚠️  Could not record {tf} analysis: {e}", err=True)
    return saved


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--record/--no-record",
    default=False,
    help="Save each timeframe's regime to the history database ($MRA_DB_PATH)",
)
@provider_options
@click.pass_context
@handle_exceptions
def current_analysis(
    ctx: click.Context, symbol: str, record: bool, provider: str | None, api_key: str | None
) -> None:
    """Run current HMM regime analysis for all timeframes.

    Each timeframe is loaded and analyzed independently, so one failing timeframe
    does not hide the others. Exits non-zero only if every timeframe fails.

    With --record, one record per analyzed timeframe is saved to the regime
    history database (see 'mra history'); a failed save is reported but does
    not fail the command.
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
    succeeded: dict[str, RegimeAnalysis] = {}
    to_record: list[tuple[str, MarketRegimeAnalyzer, RegimeAnalysis]] = []
    for tf in TIMEFRAMES:
        result = results[tf]
        if isinstance(result, Exception):
            failed += 1
            click.echo(f"\n❌ Error analyzing {tf}: {result}", err=True)
        else:
            analyzer, analysis = result
            succeeded[tf] = analysis
            to_record.append((tf, analyzer, analysis))
            print_regime_report(symbol, tf, analysis, _last_close(analyzer, tf))

    if record and to_record:
        store = default_store()
        saved = _record_analyses(store, to_record)
        if saved:
            click.echo(f"\n🗄️  Recorded {saved} analysis record(s) to {store.path}")

    if failed == len(TIMEFRAMES):
        raise click.ClickException(f"Analysis failed for every timeframe of {symbol}")

    # Multi-timeframe confirmation: only meaningful with two or more timeframes
    if len(succeeded) >= MIN_CONFIRMATION_TIMEFRAMES:
        confirmation = confirm_timeframes(succeeded)
        click.echo(format_confirmation_report(confirmation, symbol, succeeded))


def _record_to_dict(rec: RegimeRecord) -> dict[str, Any]:
    return {
        "symbol": rec.symbol,
        "timeframe": rec.timeframe,
        "bar_time": rec.bar_time.isoformat(),
        "recorded_at": rec.recorded_at.isoformat(),
        "regime": rec.regime,
        "confidence": rec.confidence,
        "persistence": rec.persistence,
        "transition_probability": rec.transition_probability,
        "recommended_strategy": rec.recommended_strategy,
        "close": rec.close,
        "provider": rec.provider,
    }


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option(
    "--timeframe", type=TIMEFRAME_CHOICE, default=None, help="Only this timeframe (default: all)"
)
@click.option(
    "--limit",
    type=click.IntRange(1, MAX_HISTORY_LIMIT),
    default=20,
    show_default=True,
    help="Maximum number of records (newest first)",
)
@click.option(
    "--json", "as_json", is_flag=True, default=False, help="Print JSON instead of a table"
)
@handle_exceptions
def history(symbol: str, timeframe: str | None, limit: int, as_json: bool) -> None:
    """Show recorded regime history for a symbol, newest bar first.

    Records are written by 'mra current-analysis --record' to the database at
    $MRA_DB_PATH (default ~/.mra/regimes.db).
    """
    import json  # noqa: PLC0415

    if not symbol.strip():
        raise click.BadParameter("Symbol cannot be empty", param_hint="--symbol")
    records = default_store().history(symbol, timeframe=timeframe, limit=limit)

    if as_json:
        click.echo(json.dumps(_json_safe([_record_to_dict(r) for r in records]), allow_nan=False))
        return

    scope = f"{symbol.strip().upper()} ({timeframe or 'all timeframes'})"
    if not records:
        click.echo(f"No recorded regime history for {scope}.")
        return

    click.echo(f"Regime history for {scope}, newest first:")
    header = (
        f"{'Bar time (UTC)':<20} {'TF':<4} {'Regime':<16} {'Conf':>6} {'Pers':>6} "
        f"{'Strategy':<22} {'Close':>10}"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for rec in records:
        close = f"{rec.close:.2f}" if rec.close is not None else "-"
        click.echo(
            f"{rec.bar_time.strftime('%Y-%m-%d %H:%M'):<20} {rec.timeframe:<4} "
            f"{rec.regime:<16} {rec.confidence:>6.1%} {rec.persistence:>6.1%} "
            f"{rec.recommended_strategy:<22} {close:>10}"
        )


_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}


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

    # Nobody would see a figure on a non-interactive backend: save it instead
    if output is None and matplotlib.get_backend().lower() in _NON_INTERACTIVE_BACKENDS:
        output = Path(f"{symbol}_{timeframe}_regimes.png")

    if output is not None:
        fig = _render_chart(analyzer, symbol, timeframe, bars)
        fig.savefig(output, bbox_inches="tight")
        click.echo(f"✓ Chart saved to {output}")
        return

    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig = plt.figure(figsize=(15, 20))
    try:
        _render_chart(analyzer, symbol, timeframe, bars, figure=fig)
        plt.show()
    finally:
        plt.close(fig)


def _render_chart(
    analyzer: MarketRegimeAnalyzer, symbol: str, timeframe: str, bars: int, **kwargs: Any
) -> Any:
    """Draw the regime chart, turning library errors into a CLI failure."""
    try:
        return analyzer.render_regime_chart(timeframe, bars, **kwargs)
    except Exception as e:
        raise click.ClickException(
            f"Chart generation failed for {symbol} ({timeframe}): {e}"
        ) from e


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
    try:
        result: Any = analyzer.export_analysis_to_csv(filename)
    except InsufficientDataError as e:
        raise click.ClickException("No analysis data to export") from e

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
    if not portfolio.analyzers:  # defensive: the library raises when every symbol fails
        raise click.ClickException("No symbol could be loaded")
    click.echo(portfolio.format_portfolio_summary(timeframe))


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


# ── scan: scheduled watchlist scanner with regime-change alerts ──────────────


def _split_csv(value: str, option: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise click.BadParameter("must list at least one value", param_hint=option)
    return items


def _parse_timeframes(value: str, option: str) -> tuple[str, ...]:
    items = _split_csv(value, option)
    unknown = [tf for tf in items if tf not in TIMEFRAMES]
    if unknown:
        raise click.BadParameter(
            f"unknown timeframe(s) {', '.join(unknown)}; choose from {', '.join(TIMEFRAMES)}",
            param_hint=option,
        )
    return tuple(dict.fromkeys(items))


@cli.command()
@click.option(
    "--symbols", type=str, default="SPY", show_default=True, help="Comma-separated watchlist"
)
@click.option(
    "--timeframes",
    type=str,
    default=",".join(TIMEFRAMES),
    show_default=True,
    help="Comma-separated timeframes to analyze",
)
@click.option(
    "--interval",
    type=click.IntRange(min=1),
    default=3600,
    show_default=True,
    help="Seconds between scans",
)
@click.option("--once", is_flag=True, help="Run a single scan and exit")
@click.option(
    "--max-iterations",
    type=click.IntRange(min=1),
    default=None,
    help="Stop after this many scans (default: run until Ctrl+C / SIGTERM)",
)
@click.option(
    "--watch",
    type=str,
    default=",".join(DEFAULT_WATCH_TIMEFRAMES),
    show_default=True,
    help="Comma-separated timeframes whose regime changes alert",
)
@click.option(
    "--no-confirmation",
    is_flag=True,
    help="Alert without requiring multi-timeframe confirmation",
)
@click.option(
    "--min-confidence",
    type=float,
    default=DEFAULT_MIN_CONFIDENCE,
    show_default=True,
    callback=validate_percentage,
    help="Minimum regime confidence (0-1) for an alert",
)
@click.option(
    "--cooldown-bars",
    type=click.IntRange(min=0),
    default=DEFAULT_COOLDOWN_BARS,
    show_default=True,
    help="Ignore a change when the previous regime lasted at most this many bars (0 = off)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Log alerts only (no webhook/Telegram/Discord); records are still saved",
)
@provider_options
@click.pass_context
@handle_exceptions
def scan(  # noqa: PLR0913, PLR0917
    ctx: click.Context,
    symbols: str,
    timeframes: str,
    interval: int,
    once: bool,
    max_iterations: int | None,
    watch: str,
    no_confirmation: bool,
    min_confidence: float,
    cooldown_bars: int,
    dry_run: bool,
    provider: str | None,
    api_key: str | None,
) -> None:
    """Scan a watchlist on a schedule, record regimes, and alert on regime changes.

    Every scan analyzes each symbol on each timeframe, saves the results to the
    regime history database ($MRA_DB_PATH, see 'mra history'), computes the
    multi-timeframe confirmation, and sends an alert when a watched timeframe's
    regime changes on a new bar (the first scan of a symbol only records a
    baseline).

    Alerts go to the log plus every channel configured in the environment:
    ALERT_WEBHOOK_URL, TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID, DISCORD_WEBHOOK_URL.
    --dry-run sends alerts to the log only, but still saves records, so a change
    seen by a dry run is not alerted again later from the same database.

    Exits non-zero if every symbol failed (--once) or no scan succeeded.

    \b
    Examples:
        uv run mra scan --provider mock --once --dry-run
        uv run mra scan --symbols SPY,QQQ,IWM --interval 3600
    """
    symbol_list = _split_csv(symbols, "--symbols")
    timeframe_list = _parse_timeframes(timeframes, "--timeframes")
    watch_list = _parse_timeframes(watch, "--watch")
    not_scanned = [tf for tf in watch_list if tf not in timeframe_list]
    if not_scanned:
        raise click.BadParameter(
            f"{', '.join(not_scanned)} is not in --timeframes", param_hint="--watch"
        )
    provider_name, key = resolve_provider(ctx, provider, api_key)

    notifiers: list[Notifier] = [LogNotifier()]
    if not dry_run:
        notifiers += notifiers_from_env()
    policy = AlertPolicy(
        timeframes_to_watch=watch_list,
        require_confirmation=not no_confirmation,
        min_confidence=min_confidence,
        cooldown_bars=cooldown_bars,
    )
    store = default_store()
    scanner = Scanner(
        symbol_list,
        store=store,
        timeframes=timeframe_list,
        provider=provider_name,
        api_key=key,
        notifiers=notifiers,
        policy=policy,
    )

    limit = 1 if once else max_iterations
    channels = ", ".join(describe_notifier(n) for n in notifiers)
    click.echo(
        f"Alert policy: watch {', '.join(policy.timeframes_to_watch)} | confirmation "
        + ("off" if no_confirmation else "required")
        + f" | min confidence {min_confidence:.0%} | cooldown {cooldown_bars} bar(s)"
        + f"\nAlerts: {channels}{' (dry run)' if dry_run else ''} | records: {store.path}"
        + ("" if limit == 1 else "\nPress Ctrl+C to stop")
    )

    def report(scan_report: ScanReport) -> None:
        click.echo(format_scan_report(scan_report))

    # Per-timeframe "Loading data..." progress is noise in a scanner; --debug keeps it
    analyzer_logger = logging.getLogger("mra_lib.analyzer")
    previous_level = analyzer_logger.level
    if not _debug_enabled():
        analyzer_logger.setLevel(logging.WARNING)
    try:
        summary = scanner.run(interval, max_iterations=limit, on_report=report)
    finally:
        analyzer_logger.setLevel(previous_level)

    if summary.successful_iterations == 0 and summary.failed_iterations > 0:
        raise click.ClickException(
            "Every symbol failed" if limit == 1 else "Scanner finished without a successful scan"
        )


@cli.command()
@click.option(
    "--host",
    default="127.0.0.1",
    envvar="API_HOST",
    show_default=True,
    help="API server host (env API_HOST; use 0.0.0.0 to listen on all interfaces)",
)
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=8000,
    envvar="API_PORT",
    show_default=True,
    help="API server port (env API_PORT)",
)
@click.option(
    "--dev/--no-dev",
    default=False,
    help="Development mode: ENVIRONMENT=development (no JWT_SECRET needed), auto-reload",
)
@handle_exceptions
def start_api(host: str, port: int, dev: bool) -> None:
    """Start the REST API server for web access.

    \b
    Examples:
        uv run mra start-api --dev              # Development mode (localhost)
        uv run mra start-api --host 0.0.0.0     # Listen on all interfaces
    """
    try:
        from mra_web.server import serve  # noqa: PLC0415
    except ImportError:
        raise click.ClickException(
            "API server dependencies not available. Install with: uv sync"
        ) from None

    # Same entry point as ``mra-api``: --dev sets ENVIRONMENT=development so the
    # server starts without JWT_SECRET; otherwise the config is validated first.
    serve(host=host, port=port, dev=dev)


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
    if result.total_trades > 0:
        click.echo(result.format_report())

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


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--symbol", type=str, default="SPY", help="Trading symbol")
@click.option("--timeframe", type=TIMEFRAME_CHOICE, default="1D", help="Timeframe")
@click.option(
    "--period",
    type=str,
    default=None,
    help="History to load, e.g. 2y, 5y, max (default: 5y for 1D, the timeframe default otherwise)",
)
@click.option(
    "--mode",
    type=click.Choice(["walk-forward", "simple"]),
    default="walk-forward",
    show_default=True,
    help="walk-forward: out-of-sample (HMM refit on past data only); "
    "simple: one HMM fit on the whole period (IN-SAMPLE, optimistic)",
)
@click.option(
    "--params",
    "params_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON strategy parameters: a flat object or an mra-optimize output file "
    "(best_params). Default: built-in defaults",
)
@click.option(
    "--capital",
    type=click.FloatRange(min=0, min_open=True),
    default=100000.0,
    show_default=True,
    help="Initial capital",
)
@click.option(
    "--cost-model",
    type=click.Choice(["equity", "retail", "futures", "hft", "none"]),
    default="equity",
    show_default=True,
    help="Transaction cost preset",
)
@click.option(
    "--train-bars",
    type=click.IntRange(min=1),
    default=252,
    show_default=True,
    help="Walk-forward minimum training window (bars)",
)
@click.option(
    "--test-bars",
    type=click.IntRange(min=1),
    default=63,
    show_default=True,
    help="Walk-forward test window (bars)",
)
@click.option(
    "--retrain-every",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Bars between HMM refits inside a walk-forward test window",
)
@click.option(
    "--n-states", type=click.IntRange(2, 12), default=4, show_default=True, help="HMM states"
)
@click.option("--json", "json_out", is_flag=True, help="Print the report as a JSON object")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Write the trades to this CSV file",
)
@provider_options
@click.pass_context
@handle_exceptions
def backtest(  # noqa: PLR0913, PLR0917
    ctx: click.Context,
    symbol: str,
    timeframe: str,
    period: str | None,
    mode: str,
    params_file: Path | None,
    capital: float,
    cost_model: str,
    train_bars: int,
    test_bars: int,
    retrain_every: int,
    n_states: int,
    json_out: bool,
    output: Path | None,
    provider: str | None,
    api_key: str | None,
) -> None:
    """Backtest one strategy parameter set against buy-and-hold.

    \b
    walk-forward (default) reports stitched OUT-OF-SAMPLE metrics: each test
    window's HMM is fitted only on earlier bars. simple fits once on the whole
    period and is labelled IN-SAMPLE. HMM defaults match mra-optimize, so its
    output file can be passed straight to --params.

    \b
    Examples:
        mra backtest --provider mock --symbol SPY
        mra backtest --symbol SPY --params optimization_results.json
        mra backtest --symbol QQQ --mode simple --json
    """
    import json  # noqa: PLC0415

    from mra_lib.backtesting import load_strategy_params, run_backtest  # noqa: PLC0415
    from mra_lib.errors import InvalidParametersError  # noqa: PLC0415

    try:
        params = load_strategy_params(params_file) if params_file is not None else None
    except InvalidParametersError as e:
        raise click.ClickException(f"--params {params_file}: {e}") from e

    provider_name, key = resolve_provider(ctx, provider, api_key)
    period = period or ("5y" if timeframe == "1D" else DEFAULT_PERIODS[timeframe])
    click.echo(
        f"Fetching {period} of {timeframe} data for {symbol} via {provider_name}...",
        err=json_out,
    )
    data_provider = MarketDataProvider.create_provider(provider_name, api_key=key)
    df = data_provider.fetch(symbol, period, TIMEFRAME_INTERVALS[timeframe])
    click.echo(f"Running {mode} backtest on {len(df)} bars...", err=json_out)

    try:
        report = run_backtest(
            df,
            params,
            mode,
            symbol=symbol,
            timeframe=timeframe,
            initial_capital=capital,
            cost_model=cost_model,
            train_bars=train_bars,
            test_bars=test_bars,
            retrain_frequency=retrain_every,
            n_hmm_states=n_states,
            verbose=not json_out,
        )
    except InsufficientDataError as e:
        hint = (
            "load more history (--period) or shrink --train-bars/--test-bars"
            if mode == "walk-forward"
            else "load more history (--period) or use fewer --n-states"
        )
        raise click.ClickException(f"Insufficient data for {symbol}: {e}. Try to {hint}.") from e
    except InvalidParametersError as e:
        raise click.ClickException(str(e)) from e

    if output is not None:
        report.trades_frame().to_csv(output, index=False)

    if json_out:
        payload = report.to_dict()
        payload["trades_file"] = str(output) if output is not None else None
        click.echo(json.dumps(payload, indent=2, allow_nan=False))
        return

    click.echo(report.format_report())
    if output is not None:
        click.echo(f"✓ {len(report.trades)} trades written to {output}")


if __name__ == "__main__":
    cli()
