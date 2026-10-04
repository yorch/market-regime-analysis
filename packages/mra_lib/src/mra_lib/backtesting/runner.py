"""
Backtest one strategy parameter set against buy-and-hold.

:func:`run_backtest` is a thin orchestration layer over the existing pieces
(:class:`WalkForwardValidator`, :class:`BacktestEngine`,
:class:`PerformanceMetrics`) so that front ends (the ``mra backtest`` command,
a future API endpoint) do not reimplement any engine, metrics or walk-forward
logic. It returns a typed :class:`BacktestReport`.

Two modes:

* ``walk-forward`` (default): the HMM for each test window is fitted only on
  earlier bars and the per-window equity curves are stitched. Results are
  **out-of-sample** with respect to the regime model.
* ``simple``: the HMM is fitted once on the whole period and the engine runs
  over that same period. Results are **in-sample** (optimistic).

Strategy parameters use the flat format of
:meth:`RegimeStrategy.from_param_vector`. :func:`load_strategy_params` also
accepts the JSON written by ``mra-optimize`` (``best_params`` key).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from mra_lib.config.regime_tables import periods_per_year as timeframe_periods_per_year
from mra_lib.errors import InsufficientDataError, InvalidParametersError
from mra_lib.indicators.true_hmm_detector import TrueHMMDetector

from .engine import BacktestEngine
from .metrics import PerformanceMetrics
from .strategy import PARAM_KEYS, RegimeStrategy
from .transaction_costs import (
    EquityCostModel,
    FuturesCostModel,
    HighFrequencyCostModel,
    RetailCostModel,
    TransactionCostModel,
)
from .walk_forward import MIN_TEST_BARS, WalkForwardValidator

logger = logging.getLogger(__name__)

#: Supported backtest modes.
BACKTEST_MODES: Final = ("walk-forward", "simple")


def _zero_cost_model() -> TransactionCostModel:
    return TransactionCostModel(
        spread_bps=0.0,
        commission_per_share=0.0,
        commission_min=0.0,
        slippage_bps=0.0,
        market_impact_coeff=0.0,
    )


#: Named transaction cost presets accepted by :func:`make_cost_model`.
COST_MODELS: Final[Mapping[str, Callable[[], TransactionCostModel]]] = {
    "equity": EquityCostModel,
    "retail": RetailCostModel,
    "futures": FuturesCostModel,
    "hft": HighFrequencyCostModel,
    "none": _zero_cost_model,
}

# Parameter type/range rules (see RegimeStrategy.from_param_vector for meanings)
_MULTIPLIER_PARAMS: Final = frozenset(
    {"bull_mult", "bear_mult", "mr_mult", "hv_mult", "lv_mult", "bo_mult"}
)
_FLAG_PARAMS: Final = frozenset({"confidence_scaling", "bear_short"})
_OPTIONAL_PARAMS: Final = frozenset({"stop_loss", "take_profit"})

#: Column order of :meth:`BacktestReport.trades_frame` (engine trade fields).
TRADE_COLUMNS: Final = (
    "entry_date",
    "exit_date",
    "direction",
    "shares",
    "entry_price",
    "exit_price",
    "entry_regime",
    "exit_regime",
    "gross_pnl",
    "entry_costs",
    "exit_costs",
    "pnl",
    "return_pct",
    "holding_days",
)


def make_cost_model(name: str) -> TransactionCostModel:
    """
    Build a named transaction cost preset.

    Args:
        name: One of :data:`COST_MODELS` (``equity``, ``retail``, ``futures``,
            ``hft``, ``none``)

    Returns:
        A new cost model instance.

    Raises:
        InvalidParametersError: If ``name`` is not a known preset.
    """
    try:
        factory = COST_MODELS[name]
    except KeyError:
        raise InvalidParametersError(
            f"Unknown cost model '{name}'. Available: {', '.join(COST_MODELS)}"
        ) from None
    return factory()


# ----------------------------------------------------------------------
# Strategy parameters
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyParams:
    """
    Validated flat strategy parameters plus where they came from.

    Attributes:
        params: Flat parameters for :meth:`RegimeStrategy.from_param_vector`
            (empty means defaults).
        source: Human-readable origin (``defaults``, ``parameter file`` or
            ``mra-optimize output``).
        selected_through: Last bar of the period the parameters were selected
            on (from an ``mra-optimize`` file), if known. Bars up to this date
            influenced parameter choice, so they are not out-of-sample with
            respect to the parameters.
    """

    params: dict[str, Any] = field(default_factory=dict)
    source: str = "defaults"
    selected_through: pd.Timestamp | None = None

    def build_strategy(self) -> RegimeStrategy:
        """Return the :class:`RegimeStrategy` these parameters describe."""
        return RegimeStrategy.from_param_vector(self.params)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _check_value(key: str, value: Any) -> str | None:  # noqa: PLR0911 - one return per rule
    """Return an error message for an invalid ``key=value``, or None if valid."""
    if key in _FLAG_PARAMS:
        if isinstance(value, bool) or (_is_number(value) and value in (0, 1)):
            return None
        return f"{key} must be true/false or 0/1, got {value!r}"
    if key in _OPTIONAL_PARAMS and value is None:
        return None
    if not _is_number(value):
        return f"{key} must be a number, got {value!r}"
    v = float(value)
    if key in _MULTIPLIER_PARAMS:
        return None if v >= 0 else f"{key} must be >= 0, got {value!r}"
    if key in {"base_fraction", "max_position"}:
        return None if 0 < v <= 1 else f"{key} must be in (0, 1], got {value!r}"
    if key == "min_confidence":
        return None if 0 <= v <= 1 else f"{key} must be in [0, 1], got {value!r}"
    if key == "stop_loss":
        return None if 0 <= v < 1 else f"{key} must be in [0, 1) or null, got {value!r}"
    if key == "take_profit":
        return None if v >= 0 else f"{key} must be >= 0 or null, got {value!r}"
    return None  # pragma: no cover - every PARAM_KEYS entry has a rule above


def validate_strategy_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate a flat strategy parameter mapping (keys, types and ranges).

    Args:
        params: Mapping of :data:`~mra_lib.backtesting.strategy.PARAM_KEYS` to values

    Returns:
        A plain ``dict`` copy of ``params``.

    Raises:
        InvalidParametersError: On unknown keys or invalid values (all problems
            are reported in one message).
    """
    if not isinstance(params, Mapping):
        raise InvalidParametersError(
            f"Strategy parameters must be a JSON object, got {type(params).__name__}"
        )
    problems: list[str] = []
    unknown = sorted(set(params) - PARAM_KEYS)
    if unknown:
        problems.append(
            f"unknown parameter(s): {', '.join(map(str, unknown))} "
            f"(valid: {', '.join(sorted(PARAM_KEYS))})"
        )
    for key in sorted(set(params) & PARAM_KEYS):
        msg = _check_value(key, params[key])
        if msg:
            problems.append(msg)
    if problems:
        raise InvalidParametersError("Invalid strategy parameters: " + "; ".join(problems))
    return dict(params)


def parse_strategy_params(data: Any, source: str = "parameter file") -> StrategyParams:
    """
    Parse decoded JSON into validated :class:`StrategyParams`.

    Accepts either a flat parameter object (``{"bull_mult": 1.5, ...}``) or the
    report written by ``mra-optimize`` (``{"best_params": {...}, "search": {...}}``),
    in which case the end of the optimizer's search period is recorded as
    ``selected_through``.

    Args:
        data: Decoded JSON value
        source: Label used when ``data`` is a flat parameter object

    Returns:
        Validated parameters.

    Raises:
        InvalidParametersError: If the structure, keys or values are invalid.
    """
    if not isinstance(data, Mapping):
        raise InvalidParametersError(
            f"Parameter file must contain a JSON object, got {type(data).__name__}"
        )
    if "best_params" not in data:
        return StrategyParams(params=validate_strategy_params(data), source=source)

    selected_through: pd.Timestamp | None = None
    search = data.get("search")
    if isinstance(search, Mapping) and search.get("search_end"):
        try:
            parsed = pd.Timestamp(search["search_end"])
        except (TypeError, ValueError):
            logger.warning("Ignoring unparsable search_end %r", search["search_end"])
        else:
            if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
                selected_through = parsed
    return StrategyParams(
        params=validate_strategy_params(data["best_params"]),
        source="mra-optimize output",
        selected_through=selected_through,
    )


def load_strategy_params(path: str | Path) -> StrategyParams:
    """
    Load and validate strategy parameters from a JSON file.

    Args:
        path: Path to a flat parameter file or an ``mra-optimize`` output file

    Returns:
        Validated parameters.

    Raises:
        InvalidParametersError: If the file is not valid JSON or the parameters
            are invalid.
        OSError: If the file cannot be read.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidParametersError(f"{path} is not valid JSON: {e}") from e
    return parse_strategy_params(data, source=f"parameter file ({Path(path).name})")


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SideMetrics:
    """
    Headline metrics for one side of the comparison (strategy or buy-and-hold).

    Ratios follow :class:`PerformanceMetrics` conventions. ``profit_factor`` may
    be ``inf`` (no losing trades) or ``nan`` (undefined).

    Attributes:
        total_return: Total return over the evaluated bars
        cagr: Annualized return
        sharpe: Sharpe ratio
        sortino: Sortino ratio
        calmar: Calmar ratio
        max_drawdown: Maximum drawdown (negative fraction)
        max_drawdown_duration: Longest drawdown, in bars
        win_rate: Fraction of winning trades
        profit_factor: Gross wins / gross losses
        n_trades: Number of closed trades
        time_in_market: Fraction of bars with an open position (at the close)
        avg_exposure: Mean position notional as a fraction of equity, over all bars
    """

    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    n_trades: int
    time_in_market: float
    avg_exposure: float

    @classmethod
    def from_performance(
        cls, perf: PerformanceMetrics, time_in_market: float, avg_exposure: float
    ) -> SideMetrics:
        """Build from a :class:`PerformanceMetrics` plus exposure figures."""
        m = perf.metrics
        return cls(
            total_return=float(m["total_return"]),
            cagr=float(m["annualized_return"]),
            sharpe=float(m["sharpe_ratio"]),
            sortino=float(m["sortino_ratio"]),
            calmar=float(m["calmar_ratio"]),
            max_drawdown=float(m["max_drawdown"]),
            max_drawdown_duration=int(m["max_drawdown_duration"]),
            win_rate=float(m["win_rate"]),
            profit_factor=float(m["profit_factor"]),
            n_trades=int(m["total_trades"]),
            time_in_market=float(time_in_market),
            avg_exposure=float(avg_exposure),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (non-finite floats become ``None``)."""
        return {k: _json_value(v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class WindowSummary:
    """One walk-forward test window (out-of-sample)."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    bars: int
    strategy_return: float
    buy_hold_return: float
    trades: int
    max_drawdown: float

    @property
    def excess_return(self) -> float:
        """Strategy return minus buy-and-hold return for this window."""
        return self.strategy_return - self.buy_hold_return

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict."""
        d = {k: _json_value(v) for k, v in self.__dict__.items()}
        d["excess_return"] = _json_value(self.excess_return)
        return d


@dataclass
class BacktestReport:
    """
    Result of :func:`run_backtest`.

    Attributes:
        symbol: Symbol label (informational)
        timeframe: Timeframe used for annualization
        mode: ``walk-forward`` or ``simple``
        in_sample: True when the regime model saw the evaluated bars (``simple``)
        start: First evaluated bar
        end: Last evaluated bar
        n_bars: Number of evaluated bars
        years: Evaluated span in years (bars / periods per year)
        initial_capital: Starting capital
        cost_model: Cost model name
        params: Strategy parameters used (empty means defaults)
        params_source: Where the parameters came from
        params_selected_through: End of the parameter search period, if known
        overlapping_windows: Walk-forward windows whose test period starts on or
            before ``params_selected_through`` (not out-of-sample w.r.t. the
            parameters)
        settings: Regime-model / walk-forward settings
        strategy: Strategy metrics
        buy_hold: Buy-and-hold metrics over the same bars
        trades: Closed trades (engine trade dicts; ``window`` added in walk-forward)
        equity_curve: Strategy equity per evaluated bar (stitched in walk-forward)
        buy_hold_curve: Buy-and-hold equity per evaluated bar
        windows: Per-window summaries (walk-forward only)
        n_fits: Successful HMM fits
        refit_failures: Failed HMM fits (the previous model was kept)
        warnings: Caveats worth showing to the user
    """

    symbol: str
    timeframe: str
    mode: str
    in_sample: bool
    start: pd.Timestamp
    end: pd.Timestamp
    n_bars: int
    years: float
    initial_capital: float
    cost_model: str
    params: dict[str, Any]
    params_source: str
    params_selected_through: pd.Timestamp | None
    overlapping_windows: int
    settings: dict[str, Any]
    strategy: SideMetrics
    buy_hold: SideMetrics
    trades: list[dict[str, Any]]
    equity_curve: pd.Series
    buy_hold_curve: pd.Series
    windows: list[WindowSummary] = field(default_factory=list)
    n_fits: int = 0
    refit_failures: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def sample_label(self) -> str:
        """``IN-SAMPLE`` or ``OUT-OF-SAMPLE``."""
        return "IN-SAMPLE" if self.in_sample else "OUT-OF-SAMPLE"

    @property
    def sample_note(self) -> str:
        """One-sentence explanation of what the sample label means here."""
        if self.in_sample:
            return (
                "The HMM was fitted on the same period it is evaluated on, so these "
                "results are optimistic and are NOT an out-of-sample estimate "
                "(use mode walk-forward for that)."
            )
        return (
            "Each test window's HMM was fitted only on earlier bars; window equity "
            "curves are stitched into one out-of-sample curve."
        )

    @property
    def excess_return(self) -> float:
        """Strategy total return minus buy-and-hold total return."""
        return self.strategy.total_return - self.buy_hold.total_return

    def trades_frame(self) -> pd.DataFrame:
        """
        Trades as a DataFrame, one row per closed trade.

        Columns follow :data:`TRADE_COLUMNS`, preceded by ``window`` in
        walk-forward mode; any extra engine fields are appended. An empty frame
        still has the columns, so a CSV export always has a header.
        """
        columns = (["window"] if self.mode == "walk-forward" else []) + list(TRADE_COLUMNS)
        frame = pd.DataFrame(self.trades)
        extra = [c for c in frame.columns if c not in columns]
        return frame.reindex(columns=columns + extra)

    def to_dict(self, include_trades: bool = False) -> dict[str, Any]:
        """
        JSON-safe summary (timestamps as ISO strings, non-finite floats as ``None``).

        Args:
            include_trades: Also include the list of trades
        """
        out: dict[str, Any] = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "mode": self.mode,
            "sample": self.sample_label.lower(),
            "in_sample": self.in_sample,
            "note": self.sample_note,
            "period": {
                "start": _json_value(self.start),
                "end": _json_value(self.end),
                "bars": self.n_bars,
                "years": _json_value(self.years),
            },
            "initial_capital": self.initial_capital,
            "cost_model": self.cost_model,
            "params": {k: _json_value(v) for k, v in self.params.items()},
            "params_source": self.params_source,
            "params_selected_through": _json_value(self.params_selected_through),
            "overlapping_windows": self.overlapping_windows,
            "settings": {k: _json_value(v) for k, v in self.settings.items()},
            "strategy": self.strategy.to_dict(),
            "buy_hold": self.buy_hold.to_dict(),
            "excess_return": _json_value(self.excess_return),
            "n_fits": self.n_fits,
            "refit_failures": self.refit_failures,
            "windows": [w.to_dict() for w in self.windows],
            "warnings": list(self.warnings),
        }
        if include_trades:
            out["trades"] = [{k: _json_value(v) for k, v in t.items()} for t in self.trades]
        return out

    def format_report(self) -> str:
        """Format the side-by-side report (strategy vs buy-and-hold) as text."""
        return format_backtest_report(self)


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------


def _json_value(value: Any) -> Any:
    """Convert numpy/pandas scalars and timestamps to JSON-safe primitives."""
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):  # datetime / date
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt_pf(pf: float) -> str:
    if math.isnan(pf):
        return "n/a"
    return "inf" if math.isinf(pf) else f"{pf:.2f}"


def _fmt_date(ts: pd.Timestamp | None) -> str:
    if ts is None:
        return "?"
    return str(ts.date()) if ts == ts.normalize() else str(ts)


def format_backtest_report(report: BacktestReport) -> str:
    """
    Format a :class:`BacktestReport` as a plain-text report.

    Args:
        report: Result of :func:`run_backtest`

    Returns:
        Multi-line report (no trailing newline).
    """
    width = 88
    s, b = report.strategy, report.buy_hold
    st = report.settings
    lines: list[str] = []
    lines.append("=" * width)
    lines.append(
        f"BACKTEST - {report.symbol or '?'} ({report.timeframe}) | {report.mode} | "
        f"{report.sample_label}"
    )
    lines.append("=" * width)
    lines.append(
        f"Period:      {_fmt_date(report.start)} .. {_fmt_date(report.end)} "
        f"({report.n_bars} bars evaluated, {report.years:.2f} years)"
    )
    if report.mode == "walk-forward":
        lines.append(
            f"Walk-fwd:    {'anchored' if st.get('anchored') else 'rolling'}, "
            f"train {st.get('train_bars')} bars, test {st.get('test_bars')} bars, "
            f"refit every {st.get('retrain_frequency')} bars, "
            f"{st.get('n_hmm_states')} HMM states"
        )
    else:
        lines.append(
            f"Model:       one HMM fit on the full period, {st.get('n_hmm_states')} states"
        )
    lines.append(f"Capital:     ${report.initial_capital:,.2f} | costs: {report.cost_model}")
    lines.append(f"Parameters:  {report.params_source}")
    if report.params:
        lines.append(
            "             "
            + ", ".join(
                f"{k}={v:g}" if _is_number(v) else f"{k}={v}"
                for k, v in sorted(report.params.items())
            )
        )
    lines.append(f"{report.sample_label}: {report.sample_note}")

    lines.append("")
    lines.append(f"{'METRIC':<26}{'STRATEGY':>14}{'BUY & HOLD':>14}")
    lines.append("-" * 54)
    rows: list[tuple[str, str, str]] = [
        ("Total Return", f"{s.total_return:+.2%}", f"{b.total_return:+.2%}"),
        ("CAGR", f"{s.cagr:+.2%}", f"{b.cagr:+.2%}"),
        ("Sharpe Ratio", f"{s.sharpe:.2f}", f"{b.sharpe:.2f}"),
        ("Sortino Ratio", f"{s.sortino:.2f}", f"{b.sortino:.2f}"),
        ("Calmar Ratio", f"{s.calmar:.2f}", f"{b.calmar:.2f}"),
        ("Max Drawdown", f"{s.max_drawdown:.2%}", f"{b.max_drawdown:.2%}"),
        (
            "Max DD Duration (bars)",
            f"{s.max_drawdown_duration}",
            f"{b.max_drawdown_duration}",
        ),
        ("Win Rate", f"{s.win_rate:.1%}", f"{b.win_rate:.1%}"),
        ("Profit Factor", _fmt_pf(s.profit_factor), _fmt_pf(b.profit_factor)),
        ("Trades", f"{s.n_trades}", f"{b.n_trades}"),
        ("Time in Market", f"{s.time_in_market:.1%}", f"{b.time_in_market:.1%}"),
        ("Avg Exposure (of equity)", f"{s.avg_exposure:.1%}", f"{b.avg_exposure:.1%}"),
    ]
    for label, sv, bv in rows:
        lines.append(f"{label:<26}{sv:>14}{bv:>14}")
    lines.append("-" * 54)
    lines.append(f"Excess return vs buy & hold: {report.excess_return:+.2%}")
    lines.append(
        "Buy & hold: fully invested over the same bars, no costs; one 'trade' per "
        + ("test window." if report.mode == "walk-forward" else "period.")
    )

    if report.windows:
        lines.append("")
        lines.append("PER-WINDOW SUMMARY (out-of-sample test windows)")
        lines.append(
            f"{'#':>3}  {'Test Start':<12}{'Test End':<12}{'Bars':>5}{'Strategy':>10}"
            f"{'B&H':>10}{'Excess':>10}{'Trades':>7}{'MaxDD':>9}"
        )
        for w in report.windows:
            lines.append(
                f"{w.index:>3}  {_fmt_date(w.test_start):<12}{_fmt_date(w.test_end):<12}"
                f"{w.bars:>5}{w.strategy_return:>+10.2%}{w.buy_hold_return:>+10.2%}"
                f"{w.excess_return:>+10.2%}{w.trades:>7}{w.max_drawdown:>9.2%}"
            )
        beating = sum(1 for w in report.windows if w.excess_return > 0)
        lines.append(
            f"Windows beating buy & hold: {beating}/{len(report.windows)} | "
            f"HMM fits: {report.n_fits} | refit failures: {report.refit_failures}"
        )

    if report.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        lines.extend(f"  - {w}" for w in report.warnings)
    lines.append("=" * width)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def _exposure(
    trades: list[dict[str, Any]], closes: pd.Series, equity: pd.Series
) -> tuple[int, float]:
    """
    Bars held and summed per-bar exposure for one engine run.

    A position counts as held at the close of bars ``entry .. exit-1`` (it is
    opened at the entry bar's close and gone by the exit bar's close).

    Returns:
        ``(held_bars, sum of position notional / equity over held bars)``
    """
    index = equity.index
    held = 0
    exposure_sum = 0.0
    for t in trades:
        start = int(index.get_indexer([t["entry_date"]])[0])
        stop = int(index.get_indexer([t["exit_date"]])[0])
        if start < 0 or stop < 0 or stop <= start:
            continue
        held += stop - start
        notional = closes.iloc[start:stop].to_numpy(dtype=float) * float(t["shares"])
        eq = equity.iloc[start:stop].to_numpy(dtype=float)
        exposure_sum += float(np.sum(np.where(eq > 0, notional / np.where(eq > 0, eq, 1.0), 0.0)))
    return held, exposure_sum


def _buy_hold(
    segments: list[pd.Series], initial_capital: float
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """
    Chained buy-and-hold equity over ``segments`` of closes, plus one trade each.

    Each segment is held from its first close to its last close (matching the
    per-window benchmark of :class:`WalkForwardValidator`), with no costs.
    """
    pieces: list[pd.Series] = []
    trades: list[dict[str, Any]] = []
    level = initial_capital
    for segment in segments:
        closes = segment.astype(float)
        if len(closes) == 0 or closes.iloc[0] <= 0:
            continue
        curve = level * closes / closes.iloc[0]
        trades.append(
            {
                "entry_date": closes.index[0],
                "exit_date": closes.index[-1],
                "pnl": float(curve.iloc[-1] - level),
            }
        )
        pieces.append(curve)
        level = float(curve.iloc[-1])
    if not pieces:
        return pd.Series(dtype=float), trades
    return pd.concat(pieces), trades


def _to_index_tz(ts: pd.Timestamp | None, index: pd.Index) -> pd.Timestamp | None:
    """Align ``ts``'s timezone with ``index`` so the two can be compared."""
    if ts is None:
        return None
    tz = getattr(index, "tz", None)
    if tz is None:
        return ts.tz_convert(None) if ts.tzinfo is not None else ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


@dataclass
class _ModeRun:
    """What one mode produces before the shared metric computation."""

    equity: pd.Series
    trades: list[dict[str, Any]]
    segments: list[pd.Series]  # close-price segments the strategy was evaluated on
    held_bars: int
    exposure_sum: float
    n_fits: int
    refit_failures: int = 0
    windows: list[WindowSummary] = field(default_factory=list)
    overlapping: int = 0
    warnings: list[str] = field(default_factory=list)


def _check_walk_forward_windows(
    n_rows: int, train_bars: int, test_bars: int, n_states: int, min_fit_bars: int
) -> None:
    """Reject window settings the model cannot use, or too little data for them."""
    if train_bars < min_fit_bars:
        raise InvalidParametersError(
            f"train_bars={train_bars} is too small for a {n_states}-state HMM "
            f"(needs at least {min_fit_bars} bars)"
        )
    if test_bars < MIN_TEST_BARS:
        raise InvalidParametersError(f"test_bars must be at least {MIN_TEST_BARS}")
    needed = train_bars + test_bars
    if n_rows < needed:
        raise InsufficientDataError(
            f"Walk-forward needs at least {needed} bars (train {train_bars} + "
            f"test {test_bars}), got {n_rows}"
        )


def _run_walk_forward(
    validator: WalkForwardValidator,
    df: pd.DataFrame,
    selected_through: pd.Timestamp | None,
    verbose: bool,
) -> _ModeRun:
    """Walk-forward run (out-of-sample); regimes are computed once and reused."""
    cache = validator.compute_regimes(df)
    results = validator.run(df, verbose=verbose, regime_cache=cache)
    if "error" in results:
        raise InsufficientDataError(f"No valid walk-forward windows ({results['error']})")

    run = _ModeRun(
        equity=results["stitched_equity_curve"],
        trades=[],
        segments=[],
        held_bars=0,
        exposure_sum=0.0,
        n_fits=cache.fit_count,
        refit_failures=cache.refit_failures,
    )
    for i, w in enumerate(results["window_results"], start=1):
        perf: PerformanceMetrics = w["performance"]
        eq = perf.equity_curve
        closes = df["Close"].loc[eq.index]
        run.segments.append(closes)
        held, exposure = _exposure(perf.trades, closes, eq)
        run.held_bars += held
        run.exposure_sum += exposure
        run.trades.extend({**t, "window": i} for t in perf.trades)
        run.windows.append(
            WindowSummary(
                index=i,
                train_start=w["train_start"],
                train_end=w["train_end"],
                test_start=w["test_start"],
                test_end=w["test_end"],
                bars=int(w["test_days"]),
                strategy_return=float(w["strategy_return"]),
                buy_hold_return=float(w["buy_hold_return"]),
                trades=int(w["trades"]),
                max_drawdown=float(perf.metrics["max_drawdown"]),
            )
        )
        if selected_through is not None and w["test_start"] <= selected_through:
            run.overlapping += 1

    if run.refit_failures:
        run.warnings.append(
            f"{run.refit_failures} HMM refit(s) failed; the previous model was kept "
            "(bars before the first successful fit are UNKNOWN and not traded)."
        )
    if run.overlapping:
        run.warnings.append(
            f"{run.overlapping} of {len(run.windows)} test windows start on or before "
            f"{_fmt_date(selected_through)}, the end of the period the parameters "
            "were selected on: those windows are out-of-sample for the regime model "
            "but NOT for the parameters."
        )
    return run


def _run_simple(
    validator: WalkForwardValidator,
    df: pd.DataFrame,
    selected_through: pd.Timestamp | None,
) -> _ModeRun:
    """Single fit on the whole period, then one engine run over it (in-sample)."""
    strategy = validator.strategy
    regimes, confidences = validator.detect_regimes_in_sample(df)
    strategies, directions, position_sizes = strategy.generate_signals(regimes, confidences)
    # Same engine configuration as WalkForwardValidator uses for each test window
    engine = BacktestEngine(
        initial_capital=validator.initial_capital,
        cost_model=validator.cost_model,
        max_position_size=strategy.max_position_size,
        stop_loss_pct=strategy.stop_loss_pct,
        take_profit_pct=strategy.take_profit_pct,
        periods_per_year=validator.periods_per_year,
    )
    res = engine.run_regime_strategy(
        df=df,
        regimes=regimes,
        strategies=strategies,
        position_sizes=position_sizes,
        directions=directions,
    )
    trades = list(res["trades"])
    held, exposure = _exposure(trades, df["Close"], res["equity_curve"])
    run = _ModeRun(
        equity=res["equity_curve"],
        trades=trades,
        segments=[df["Close"]],
        held_bars=held,
        exposure_sum=exposure,
        n_fits=1,
    )
    if selected_through is not None and df.index[0] <= selected_through:
        run.warnings.append(
            f"The parameters were selected on data through {_fmt_date(selected_through)}, "
            "which overlaps this period."
        )
    return run


def run_backtest(
    df: pd.DataFrame,
    params: StrategyParams | Mapping[str, Any] | None = None,
    mode: str = "walk-forward",
    *,
    symbol: str = "",
    timeframe: str = "1D",
    initial_capital: float = 100000.0,
    cost_model: str = "equity",
    train_bars: int = 252,
    test_bars: int = 63,
    retrain_frequency: int = 20,
    n_hmm_states: int = 4,
    hmm_n_iter: int = 50,
    anchored: bool = True,
    risk_free_rate: float = 0.02,
    verbose: bool = False,
) -> BacktestReport:
    """
    Backtest one strategy parameter set and compare it with buy-and-hold.

    The HMM defaults (4 states, 50 EM iterations, refit every 20 bars, 252/63
    bar windows, anchored) match ``mra-optimize``, so a parameter set picked by
    the optimizer is evaluated with the same regime model.

    Args:
        df: OHLCV DataFrame (``Open``/``High``/``Low``/``Close``, optional ``Volume``)
        params: Strategy parameters (validated mapping, :class:`StrategyParams`,
            or None for defaults)
        mode: ``walk-forward`` (out-of-sample) or ``simple`` (in-sample)
        symbol: Symbol label for the report
        timeframe: Timeframe used for annualization (e.g. ``1D``, ``1H``, ``15m``)
        initial_capital: Starting capital
        cost_model: Cost preset name (see :data:`COST_MODELS`)
        train_bars: Walk-forward minimum training window, in bars
        test_bars: Walk-forward test window, in bars
        retrain_frequency: Bars between HMM refits within a test window
        n_hmm_states: Number of HMM states
        hmm_n_iter: Max EM iterations per fit
        anchored: Anchored (growing) rather than rolling training window
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino
        verbose: Log walk-forward progress at INFO level

    Returns:
        The backtest report.

    Raises:
        InvalidParametersError: On an unknown mode/cost model, invalid strategy
            parameters, or window settings the model cannot be fitted with.
        InsufficientDataError: If ``df`` is too short for the requested windows.
    """
    if mode not in BACKTEST_MODES:
        raise InvalidParametersError(
            f"Unknown mode '{mode}'. Available: {', '.join(BACKTEST_MODES)}"
        )
    if not initial_capital > 0:
        raise InvalidParametersError("initial_capital must be positive")
    if isinstance(params, StrategyParams):
        sp = params
    elif params is None:
        sp = StrategyParams()
    else:
        sp = StrategyParams(params=validate_strategy_params(params), source="parameters")

    try:
        ppy = timeframe_periods_per_year(timeframe)
    except ValueError as e:
        raise InvalidParametersError(str(e)) from e
    costs = make_cost_model(cost_model)
    strategy = sp.build_strategy()
    min_fit_bars = TrueHMMDetector(n_states=n_hmm_states).min_training_bars

    validator = WalkForwardValidator(
        strategy=strategy,
        cost_model=costs,
        n_hmm_states=n_hmm_states,
        hmm_n_iter=hmm_n_iter,
        retrain_frequency=retrain_frequency,
        min_train_days=train_bars,
        test_days=test_bars,
        anchored=anchored,
        initial_capital=initial_capital,
        periods_per_year=ppy,
        risk_free_rate=risk_free_rate,
    )
    settings: dict[str, Any] = {"n_hmm_states": n_hmm_states, "hmm_n_iter": hmm_n_iter}
    selected_through = _to_index_tz(sp.selected_through, df.index)

    if mode == "walk-forward":
        _check_walk_forward_windows(len(df), train_bars, test_bars, n_hmm_states, min_fit_bars)
        settings.update(
            train_bars=train_bars,
            test_bars=test_bars,
            retrain_frequency=retrain_frequency,
            anchored=anchored,
        )
        run = _run_walk_forward(validator, df, selected_through, verbose)
    else:
        needed = min_fit_bars + MIN_TEST_BARS
        if len(df) < needed:
            raise InsufficientDataError(
                f"Simple mode needs at least {needed} bars for a {n_hmm_states}-state HMM, "
                f"got {len(df)}"
            )
        run = _run_simple(validator, df, selected_through)

    # Every metric comes from PerformanceMetrics (same conventions as the engine)
    equity = run.equity
    strat_perf = PerformanceMetrics(
        run.trades,
        equity,
        risk_free_rate=risk_free_rate,
        periods_per_year=ppy,
        initial_capital=initial_capital,
    )
    bh_curve, bh_trades = _buy_hold(run.segments, initial_capital)
    bh_perf = PerformanceMetrics(
        bh_trades,
        bh_curve,
        risk_free_rate=risk_free_rate,
        periods_per_year=ppy,
        initial_capital=initial_capital,
    )
    n_bars = len(equity)
    strat_side = SideMetrics.from_performance(
        strat_perf,
        time_in_market=run.held_bars / n_bars if n_bars else 0.0,
        avg_exposure=run.exposure_sum / n_bars if n_bars else 0.0,
    )
    bh_side = SideMetrics.from_performance(bh_perf, time_in_market=1.0, avg_exposure=1.0)
    if strat_side.n_trades == 0:
        run.warnings.append("The strategy made no trades over the evaluated period.")

    return BacktestReport(
        symbol=symbol,
        timeframe=timeframe,
        mode=mode,
        in_sample=mode == "simple",
        start=equity.index[0],
        end=equity.index[-1],
        n_bars=n_bars,
        years=float(strat_perf.metrics["years"]),
        initial_capital=initial_capital,
        cost_model=cost_model,
        params=dict(sp.params),
        params_source=sp.source,
        params_selected_through=selected_through,
        overlapping_windows=run.overlapping,
        settings=settings,
        strategy=strat_side,
        buy_hold=bh_side,
        trades=run.trades,
        equity_curve=equity,
        buy_hold_curve=bh_curve,
        windows=run.windows,
        n_fits=run.n_fits,
        refit_failures=run.refit_failures,
        warnings=run.warnings,
    )
