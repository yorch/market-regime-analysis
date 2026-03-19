"""
Empirical regime multiplier calibration.

Analyzes per-regime trade performance from historical walk-forward
backtest results and derives optimal position size multipliers.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ..enums import MarketRegime
from .strategy import RegimeStrategy
from .transaction_costs import EquityCostModel, TransactionCostModel
from .walk_forward import WalkForwardValidator

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
        """
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

    def _collect_trades(
        self,
        strategy: RegimeStrategy,
        verbose: bool = False,
    ) -> tuple[list[dict], float]:
        """
        Run walk-forward and collect all trades.

        Returns:
            Tuple of (trades_list, baseline_sharpe)
        """
        validator = WalkForwardValidator(
            strategy=strategy,
            cost_model=self.cost_model,
            n_hmm_states=self.n_hmm_states,
            hmm_n_iter=self.hmm_n_iter,
            retrain_frequency=self.retrain_frequency,
            min_train_days=self.min_train_days,
            test_days=self.test_days,
            anchored=self.anchored,
            initial_capital=self.initial_capital,
        )

        wf_results = validator.run(self.df, verbose=verbose)

        if "error" in wf_results:
            return [], 0.0

        # Collect all trades from all windows
        all_trades = []
        for w in wf_results.get("window_results", []):
            all_trades.extend(w["performance"].trades)

        sharpe = wf_results.get("sharpe_approx", 0.0)
        return all_trades, sharpe

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
            rs.std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 0.0

            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]

            rs.win_rate = len(winners) / len(pnls) if pnls else 0.0
            rs.avg_win = float(np.mean(winners)) if winners else 0.0
            rs.avg_loss = float(np.mean([abs(x) for x in losers])) if losers else 0.0

            total_wins = sum(winners) if winners else 0.0
            total_losses = sum(abs(x) for x in losers) if losers else 0.0
            rs.profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

            # Per-regime Sharpe (annualized approximation)
            holding_days = [max(t.get("holding_days", 1), 1) for t in rtrades]
            rs.avg_holding_days = float(np.mean(holding_days))
            if rs.std_pnl > 0 and rs.avg_holding_days > 0:
                trades_per_year = 252.0 / rs.avg_holding_days
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
                # Excess over breakeven
                scores[regime] = max(0.0, rs.profit_factor - 1.0)
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
            verbose: Print progress

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
            verbose: Print progress

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
            print(f"  Calibrating multipliers using '{method}' method...")
            print("  Running walk-forward with uniform multipliers...")

        # Collect trades
        trades, baseline_sharpe = self._collect_trades(base_strategy, verbose=verbose)

        if not trades:
            if verbose:
                print("  WARNING: No trades collected. Returning default multipliers.")
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

        if verbose:
            self._print_calibration_report(regime_stats, raw_scores, multipliers, method)

        return CalibrationResult(
            multipliers=multipliers,
            regime_stats=regime_stats,
            method=method,
            total_trades=len(trades),
            trades_per_regime=trades_per_regime,
            baseline_sharpe=baseline_sharpe,
            raw_scores=raw_scores,
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
            verbose: Print progress

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

    def _print_calibration_report(
        self,
        stats: dict[MarketRegime, RegimeTradeStats],
        scores: dict[MarketRegime, float],
        multipliers: dict[MarketRegime, float],
        method: str,
    ) -> None:
        """Print formatted calibration results."""
        print("\n" + "=" * 100)
        print("REGIME MULTIPLIER CALIBRATION RESULTS")
        print(f"Method: {method}")
        print("=" * 100)

        print(
            f"\n{'Regime':<20} {'Trades':>7} {'Win%':>7} {'AvgWin':>10} "
            f"{'AvgLoss':>10} {'PF':>7} {'Sharpe':>8} {'Kelly':>7} "
            f"{'Score':>8} {'Mult':>7}"
        )
        print("-" * 100)

        for regime in MarketRegime:
            rs = stats.get(regime, RegimeTradeStats(regime=regime))
            score = scores.get(regime, 0.0)
            mult = multipliers.get(regime, 0.0)

            if rs.n_trades == 0:
                print(
                    f"{regime.value:<20} {'--':>7} {'--':>7} {'--':>10} "
                    f"{'--':>10} {'--':>7} {'--':>8} {'--':>7} "
                    f"{'--':>8} {mult:>7.2f}"
                )
            else:
                print(
                    f"{regime.value:<20} {rs.n_trades:>7} {rs.win_rate:>7.1%} "
                    f"${rs.avg_win:>9.2f} ${rs.avg_loss:>9.2f} "
                    f"{rs.profit_factor:>7.2f} {rs.sharpe:>8.2f} "
                    f"{rs.kelly_fraction:>7.2%} {score:>8.3f} {mult:>7.2f}"
                )

        print("=" * 100)

        print("\nCALIBRATED MULTIPLIERS:")
        for regime in MarketRegime:
            mult = multipliers.get(regime, 0.0)
            bar = "#" * int(mult * 10)
            print(f"  {regime.value:<20} {mult:>5.2f}  {bar}")
