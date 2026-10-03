"""
Empirical regime multiplier calibration.

Analyzes per-regime trade performance from historical walk-forward
backtest results and derives optimal position size multipliers.

By default the multipliers are fit and reported on the same data
(``CalibrationResult.in_sample`` is True). Pass ``holdout_frac > 0`` to fit on
the earlier part of the data and evaluate the calibrated strategy once on the
untouched remainder (``CalibrationResult.holdout_metrics``).

Trades are attributed to the regime at entry.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mra_lib.config.enums import MarketRegime
from mra_lib.config.regime_tables import TRADING_DAYS_PER_YEAR

from .strategy import RegimeStrategy
from .trade_stats import PROFIT_FACTOR_CAP, compute_trade_stats, finite_profit_factor
from .transaction_costs import EquityCostModel, TransactionCostModel
from .walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)

# Calibration methods
CALIBRATION_METHODS = ("sharpe_weighted", "win_rate", "profit_factor", "kelly")


@dataclass
class RegimeTradeStats:
    """Per-regime trade statistics."""

    regime: MarketRegime
    n_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    std_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    kelly_fraction: float = 0.0
    avg_holding_days: float = 0.0


@dataclass
class CalibrationResult:
    """Full result from multiplier calibration."""

    multipliers: dict[MarketRegime, float]
    regime_stats: dict[MarketRegime, RegimeTradeStats]
    method: str
    total_trades: int
    trades_per_regime: dict[MarketRegime, int]
    baseline_sharpe: float
    raw_scores: dict[MarketRegime, float] = field(default_factory=dict)
    #: True when the multipliers were fit and reported on the same data
    in_sample: bool = True
    #: Walk-forward summary of the calibrated strategy on the holdout segment
    holdout_metrics: dict | None = None

    def format_report(self) -> str:
        """
        Format the calibration table, calibrated multipliers, and the evaluation label.

        The label is the holdout (out-of-sample) evaluation when available, else a
        note that the calibration is in-sample.

        Returns:
            Multi-line report text (starts with a blank line, no trailing newline)
        """
        lines = [
            "\n" + "=" * 100,
            "REGIME MULTIPLIER CALIBRATION RESULTS (calibration period, in-sample)",
            f"Method: {self.method}",
            "=" * 100,
            f"\n{'Regime':<20} {'Trades':>7} {'Win%':>7} {'AvgWin':>10} "
            f"{'AvgLoss':>10} {'PF':>7} {'Sharpe':>8} {'Kelly':>7} "
            f"{'Score':>8} {'Mult':>7}",
            "-" * 100,
        ]

        for regime in MarketRegime:
            rs = self.regime_stats.get(regime, RegimeTradeStats(regime=regime))
            score = self.raw_scores.get(regime, 0.0)
            mult = self.multipliers.get(regime, 0.0)

            if rs.n_trades == 0:
                lines.append(
                    f"{regime.value:<20} {'--':>7} {'--':>7} {'--':>10} "
                    f"{'--':>10} {'--':>7} {'--':>8} {'--':>7} "
                    f"{'--':>8} {mult:>7.2f}"
                )
            else:
                lines.append(
                    f"{regime.value:<20} {rs.n_trades:>7} {rs.win_rate:>7.1%} "
                    f"${rs.avg_win:>9.2f} ${rs.avg_loss:>9.2f} "
                    f"{rs.profit_factor:>7.2f} {rs.sharpe:>8.2f} "
                    f"{rs.kelly_fraction:>7.2%} {score:>8.3f} {mult:>7.2f}"
                )

        lines.append("=" * 100)

        lines.append("\nCALIBRATED MULTIPLIERS:")
        for regime in MarketRegime:
            mult = self.multipliers.get(regime, 0.0)
            bar = "#" * int(mult * 10)
            lines.append(f"  {regime.value:<20} {mult:>5.2f}  {bar}")

        holdout = self.holdout_metrics
        if holdout is None:
            lines.append(
                "\nNOTE: IN-SAMPLE calibration - multipliers were fit and scored on the "
                "same data; expect weaker results out of sample."
            )
        else:
            pf = holdout["profit_factor"]
            lines += [
                "\nHOLDOUT (out-of-sample) evaluation of calibrated multipliers:",
                f"  Return:        {holdout['compounded_strategy_return']:+.2%}",
                f"  Buy & Hold:    {holdout['compounded_bh_return']:+.2%}",
                f"  Sharpe:        {holdout['sharpe_ratio']:.2f}",
                f"  Max Drawdown:  {holdout['max_drawdown']:.2%}",
                f"  Trades:        {holdout['total_trades']}",
                f"  Profit Factor: {pf:.2f}" if not math.isnan(pf) else "  Profit Factor: n/a",
            ]
        return "\n".join(lines)


class RegimeMultiplierCalibrator:
    """
    Calibrate regime multipliers empirically from historical backtest data.

    Runs a walk-forward backtest with uniform multipliers (all 1.0), then
    analyzes per-regime trade performance to derive optimal multipliers
    based on realized Sharpe, win rate, profit factor, or Kelly fraction.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cost_model: TransactionCostModel | None = None,
        n_hmm_states: int = 4,
        hmm_n_iter: int = 50,
        retrain_frequency: int = 20,
        min_train_days: int = 252,
        test_days: int = 63,
        anchored: bool = True,
        initial_capital: float = 100000.0,
        min_trades_per_regime: int = 5,
        periods_per_year: float = TRADING_DAYS_PER_YEAR,
        holdout_frac: float = 0.0,
    ) -> None:
        """
        Initialize calibrator.

        Args:
            df: Full OHLCV DataFrame
            cost_model: Transaction cost model
            n_hmm_states: Number of HMM states
            hmm_n_iter: Max HMM training iterations
            retrain_frequency: Days between HMM retrains
            min_train_days: Minimum training window size
            test_days: Test window size
            anchored: Anchored walk-forward
            initial_capital: Starting capital
            min_trades_per_regime: Minimum trades needed for reliable stats
            periods_per_year: Bars per year used for annualization
            holdout_frac: Fraction of bars at the end of ``df`` withheld from
                calibration and used once to evaluate the calibrated strategy
        """
        if not 0.0 <= holdout_frac < 1.0:
            raise ValueError("holdout_frac must be in [0, 1)")
        self.df = df
        self.cost_model = cost_model or EquityCostModel()
        self.n_hmm_states = n_hmm_states
        self.hmm_n_iter = hmm_n_iter
        self.retrain_frequency = retrain_frequency
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.anchored = anchored
        self.initial_capital = initial_capital
        self.min_trades_per_regime = min_trades_per_regime
        self.periods_per_year = periods_per_year
        self.holdout_frac = holdout_frac
        self.split_index = len(df) - round(len(df) * holdout_frac)

    def _collect_trades(
        self,
        strategy: RegimeStrategy,
        verbose: bool = False,
    ) -> tuple[list[dict], float]:
        """
        Run walk-forward on the calibration (non-holdout) period and collect trades.

        Returns:
            Tuple of (trades_list, baseline_sharpe)
        """
        validator = self._make_validator(strategy)
        wf_results = validator.run(self.df.iloc[: self.split_index], verbose=verbose)

        if "error" in wf_results:
            return [], 0.0

        # Collect all trades from all windows
        all_trades = []
        for w in wf_results.get("window_results", []):
            all_trades.extend(w["performance"].trades)

        sharpe = wf_results.get("sharpe_approx", 0.0)
        return all_trades, sharpe

    def _make_validator(self, strategy: RegimeStrategy) -> WalkForwardValidator:
        return WalkForwardValidator(
            strategy=strategy,
            cost_model=self.cost_model,
            n_hmm_states=self.n_hmm_states,
            hmm_n_iter=self.hmm_n_iter,
            retrain_frequency=self.retrain_frequency,
            min_train_days=self.min_train_days,
            test_days=self.test_days,
            anchored=self.anchored,
            initial_capital=self.initial_capital,
            periods_per_year=self.periods_per_year,
        )

    def evaluate_holdout(self, strategy: RegimeStrategy, verbose: bool = False) -> dict | None:
        """Run ``strategy`` once on the holdout segment (None if no holdout)."""
        if self.split_index >= len(self.df):
            return None
        wf = self._make_validator(strategy).run(
            self.df, verbose=verbose, start_index=self.split_index
        )
        if "error" in wf:
            return None
        return {
            k: wf[k]
            for k in (
                "n_windows",
                "total_test_days",
                "compounded_strategy_return",
                "compounded_bh_return",
                "excess_return",
                "sharpe_ratio",
                "max_drawdown",
                "total_trades",
                "trade_win_rate",
                "profit_factor",
            )
        }

    def _compute_regime_stats(
        self,
        trades: list[dict],
    ) -> dict[MarketRegime, RegimeTradeStats]:
        """Compute per-regime statistics from trade list."""
        # Group trades by entry regime
        regime_trades: dict[MarketRegime, list[dict]] = {r: [] for r in MarketRegime}

        for trade in trades:
            regime_str = trade.get("entry_regime", "Unknown")
            try:
                regime = MarketRegime(regime_str)
            except ValueError:
                regime = MarketRegime.UNKNOWN
            regime_trades[regime].append(trade)

        stats: dict[MarketRegime, RegimeTradeStats] = {}
        for regime, rtrades in regime_trades.items():
            rs = RegimeTradeStats(regime=regime, n_trades=len(rtrades))

            if not rtrades:
                stats[regime] = rs
                continue

            pnls = [t["pnl"] for t in rtrades]
            rs.total_pnl = sum(pnls)
            rs.avg_pnl = float(np.mean(pnls))
            rs.std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0

            ts = compute_trade_stats(rtrades)
            rs.win_rate = float(ts["win_rate"])
            rs.avg_win = float(ts["avg_win"])
            rs.avg_loss = float(ts["avg_loss"])
            rs.profit_factor = float(ts["profit_factor"])

            # Per-regime Sharpe (annualized approximation; holding_days are
            # calendar days, so this is approximate for intraday bars)
            holding_days = [max(t.get("holding_days", 1), 1) for t in rtrades]
            rs.avg_holding_days = float(np.mean(holding_days))
            if rs.std_pnl > 0 and rs.avg_holding_days > 0:
                trades_per_year = self.periods_per_year / rs.avg_holding_days
                rs.sharpe = (rs.avg_pnl / rs.std_pnl) * np.sqrt(trades_per_year)
            else:
                rs.sharpe = 0.0

            # Kelly fraction: f* = (bp - q) / b
            if rs.avg_loss > 0:
                b = rs.avg_win / rs.avg_loss
                p = rs.win_rate
                q = 1.0 - p
                rs.kelly_fraction = max(0.0, (b * p - q) / b) if b > 0 else 0.0
            else:
                rs.kelly_fraction = 0.0

            stats[regime] = rs

        return stats

    def _score_regimes(
        self,
        stats: dict[MarketRegime, RegimeTradeStats],
        method: str,
    ) -> dict[MarketRegime, float]:
        """
        Compute raw score per regime using the selected method.

        Args:
            stats: Per-regime statistics
            method: Scoring method name

        Returns:
            Raw score per regime (higher is better)
        """
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"Unknown calibration method: {method}")

        scores: dict[MarketRegime, float] = {}

        for regime, rs in stats.items():
            # Regimes with too few trades get zero
            if rs.n_trades < self.min_trades_per_regime:
                scores[regime] = 0.0
                continue

            if method == "sharpe_weighted":
                scores[regime] = max(0.0, rs.sharpe)
            elif method == "win_rate":
                # Only reward win rates above 50% (edge over random)
                scores[regime] = max(0.0, rs.win_rate - 0.5) * 2.0
            elif method == "profit_factor":
                # Excess over breakeven (inf -> capped, nan -> 0)
                scores[regime] = max(
                    0.0, finite_profit_factor(rs.profit_factor, PROFIT_FACTOR_CAP) - 1.0
                )
            elif method == "kelly":
                scores[regime] = max(0.0, rs.kelly_fraction)
            else:
                raise ValueError(f"Unknown calibration method: {method}")

        # UNKNOWN always gets zero
        scores[MarketRegime.UNKNOWN] = 0.0

        return scores

    def _normalize_to_multipliers(
        self,
        scores: dict[MarketRegime, float],
    ) -> dict[MarketRegime, float]:
        """
        Normalize raw scores to multipliers in [0, 2.0] range.

        Positive scores are linearly scaled so max score -> 2.0
        and min positive score -> 0.5. Zero scores stay at 0.0.
        """
        positive_scores = {r: s for r, s in scores.items() if s > 0}

        if not positive_scores:
            # No regime has positive score — return all zeros
            return dict.fromkeys(scores, 0.0)

        max_score = max(positive_scores.values())
        min_score = min(positive_scores.values())

        multipliers: dict[MarketRegime, float] = {}
        for regime, score in scores.items():
            if score <= 0:
                multipliers[regime] = 0.0
            elif max_score == min_score:
                # All positive scores are equal
                multipliers[regime] = 1.0
            else:
                # Linear map: min_score -> 0.5, max_score -> 2.0
                normalized = (score - min_score) / (max_score - min_score)
                multipliers[regime] = 0.5 + normalized * 1.5

        return multipliers

    def calibrate(
        self,
        method: str = "sharpe_weighted",
        base_strategy: RegimeStrategy | None = None,
        verbose: bool = True,
    ) -> dict[MarketRegime, float]:
        """
        Run calibration and return optimized regime multipliers.

        Args:
            method: Scoring method ('sharpe_weighted', 'win_rate',
                    'profit_factor', 'kelly')
            base_strategy: Baseline strategy for collecting trades.
                          If None, uses uniform multipliers (all 1.0).
            verbose: Log progress at INFO level (``mra_lib`` logger)

        Returns:
            Dictionary mapping MarketRegime to calibrated multiplier
        """
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {CALIBRATION_METHODS}")

        result = self.calibrate_with_details(
            method=method,
            base_strategy=base_strategy,
            verbose=verbose,
        )
        return result.multipliers

    def calibrate_with_details(
        self,
        method: str = "sharpe_weighted",
        base_strategy: RegimeStrategy | None = None,
        verbose: bool = True,
    ) -> CalibrationResult:
        """
        Run calibration and return full details including per-regime stats.

        Args:
            method: Scoring method
            base_strategy: Baseline strategy (None = uniform multipliers)
            verbose: Log progress at INFO level (``mra_lib`` logger)

        Use :meth:`CalibrationResult.format_report` for a printable report.

        Returns:
            CalibrationResult with multipliers and diagnostics
        """
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {CALIBRATION_METHODS}")

        # Build baseline strategy with uniform multipliers
        if base_strategy is None:
            uniform_mults = dict.fromkeys(MarketRegime, 1.0)
            uniform_mults[MarketRegime.UNKNOWN] = 0.0
            base_strategy = RegimeStrategy(
                regime_multipliers=uniform_mults,
                base_position_fraction=0.10,
                stop_loss_pct=0.05,
                confidence_scaling=True,
            )

        if verbose:
            logger.info("  Calibrating multipliers using '%s' method...", method)
            logger.info("  Running walk-forward with uniform multipliers...")

        # Collect trades
        trades, baseline_sharpe = self._collect_trades(base_strategy, verbose=verbose)

        if not trades:
            logger.warning("No trades collected; returning default multipliers")
            return CalibrationResult(
                multipliers={r: 1.0 if r != MarketRegime.UNKNOWN else 0.0 for r in MarketRegime},
                regime_stats={r: RegimeTradeStats(regime=r) for r in MarketRegime},
                method=method,
                total_trades=0,
                trades_per_regime=dict.fromkeys(MarketRegime, 0),
                baseline_sharpe=0.0,
            )

        # Compute per-regime statistics
        regime_stats = self._compute_regime_stats(trades)

        # Score and normalize
        raw_scores = self._score_regimes(regime_stats, method)
        multipliers = self._normalize_to_multipliers(raw_scores)

        trades_per_regime = {r: rs.n_trades for r, rs in regime_stats.items()}

        holdout_metrics = None
        if self.split_index < len(self.df):
            calibrated = self._strategy_from_multipliers(multipliers, base_strategy)
            holdout_metrics = self.evaluate_holdout(calibrated)

        return CalibrationResult(
            multipliers=multipliers,
            regime_stats=regime_stats,
            method=method,
            total_trades=len(trades),
            trades_per_regime=trades_per_regime,
            baseline_sharpe=baseline_sharpe,
            raw_scores=raw_scores,
            in_sample=holdout_metrics is None,
            holdout_metrics=holdout_metrics,
        )

    @staticmethod
    def _strategy_from_multipliers(
        multipliers: dict[MarketRegime, float], base: RegimeStrategy
    ) -> RegimeStrategy:
        return RegimeStrategy(
            regime_multipliers=multipliers,
            regime_directions=base.regime_directions,
            base_position_fraction=base.base_position_fraction,
            max_position_size=base.max_position_size,
            stop_loss_pct=base.stop_loss_pct,
            take_profit_pct=base.take_profit_pct,
            min_confidence=base.min_confidence,
            confidence_scaling=base.confidence_scaling,
        )

    def create_calibrated_strategy(
        self,
        method: str = "sharpe_weighted",
        base_params: dict | None = None,
        verbose: bool = True,
    ) -> RegimeStrategy:
        """
        Convenience method: calibrate and return a ready-to-use strategy.

        Args:
            method: Scoring method
            base_params: Additional strategy parameters (stop_loss, etc.)
            verbose: Log progress at INFO level (``mra_lib`` logger)

        Returns:
            RegimeStrategy with empirically calibrated multipliers
        """
        result = self.calibrate_with_details(method=method, verbose=verbose)

        params = base_params or {}
        return RegimeStrategy(
            regime_multipliers=result.multipliers,
            base_position_fraction=params.get("base_fraction", 0.10),
            max_position_size=params.get("max_position", 0.20),
            stop_loss_pct=params.get("stop_loss", 0.05),
            take_profit_pct=params.get("take_profit"),
            min_confidence=params.get("min_confidence", 0.0),
            confidence_scaling=params.get("confidence_scaling", True),
        )
