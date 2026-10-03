"""
Performance metrics calculation for strategy evaluation.

Calculates comprehensive trading performance statistics including:
- Returns and risk metrics
- Sharpe, Sortino, Calmar ratios
- Win rate, profit factor, expectancy
- Drawdown analysis
- Trade statistics

All metrics calculated from actual trade results.
"""

import math

import numpy as np
import pandas as pd

from mra_lib.config.regime_tables import TRADING_DAYS_PER_YEAR

from .trade_stats import compute_trade_stats

#: Default number of return periods per year (daily bars); see
#: ``mra_lib.config.regime_tables.periods_per_year(timeframe)`` for intraday.
DEFAULT_PERIODS_PER_YEAR = TRADING_DAYS_PER_YEAR

#: Denominators smaller than this are treated as zero; the ratio is then 0.0.
_EPS = 1e-12

#: Cap on annualized log-growth before exponentiation (avoids OverflowError).
_MAX_LOG_GROWTH = 700.0


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from backtest results.

    Conventions:

    * Returns are simple per-period returns of the equity curve. If
      ``initial_capital`` is given it is used as the starting point, so the
      first bar's P&L (e.g. entry costs) is included.
    * ``periods_per_year`` annualizes everything (252 for daily bars; pass
      e.g. ``252 * 26`` for 15-minute bars).
    * Sharpe = mean(r - rf/ppy) / std(r) * sqrt(ppy).
    * Sortino = mean(r - rf/ppy) / DD * sqrt(ppy), where the downside deviation
      DD = sqrt(mean(min(r - rf/ppy, 0)^2)) over *all* observations.
    * Calmar = annualized return / |max drawdown| (negative for losing strategies).
    * Any ratio whose denominator is (numerically) zero is reported as 0.0.
    * Annualized return (CAGR) is -100% if equity ends at or below zero.
    * Max drawdown duration counts a trailing drawdown that has not recovered.
    * Profit factor follows :mod:`mra_lib.backtesting.trade_stats`
      (``inf`` with no losing trades, ``nan`` when undefined).
    """

    def __init__(
        self,
        trades: list[dict],
        equity_curve: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: float = TRADING_DAYS_PER_YEAR,
        initial_capital: float | None = None,
    ):
        """
        Initialize performance calculator.

        Args:
            trades: List of trade dictionaries with:
                - entry_date, exit_date
                - entry_price, exit_price
                - shares, direction ('LONG' or 'SHORT')
                - pnl (after costs)
                - return_pct
            equity_curve: Time series of portfolio value (one point per bar)
            risk_free_rate: Annual risk-free rate (default: 2%)
            periods_per_year: Return periods per year used for annualization
            initial_capital: Equity before the first bar. If None, the first
                equity point is used as the base.
        """
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        self.trades = trades
        self.equity_curve = equity_curve
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.initial_capital = initial_capital

        values = [float(v) for v in equity_curve.to_numpy()]
        if initial_capital is not None:
            values = [float(initial_capital), *values]
        self._values = np.asarray(values, dtype=float)

        if len(self._values) >= 2:
            prev = self._values[:-1]
            safe_prev = np.where(prev > 0, prev, 1.0)
            rets = np.where(prev > 0, self._values[1:] / safe_prev - 1.0, np.nan)
            self.returns = pd.Series(rets).dropna()
        else:
            self.returns = pd.Series(dtype=float)

        self.metrics = self._calculate_all_metrics()

    def _calculate_all_metrics(self) -> dict:
        """Calculate comprehensive performance metrics (each section exactly once)."""
        metrics: dict = {}
        metrics.update(self._calculate_basic_stats())
        metrics.update(self._calculate_risk_metrics())
        metrics.update(self._calculate_trade_stats())
        metrics.update(self._calculate_drawdown_metrics())
        metrics.update(
            self._calculate_ratio_metrics(
                annual_return=metrics["annualized_return"],
                max_dd=metrics["max_drawdown"],
            )
        )
        metrics.update(self._calculate_kelly_parameters(metrics))
        return metrics

    def _calculate_basic_stats(self) -> dict:
        """Calculate basic return statistics."""
        if len(self._values) < 2 or self._values[0] <= 0:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "total_trades": len(self.trades),
                "years": 0.0,
            }

        start, end = float(self._values[0]), float(self._values[-1])
        total_return = end / start - 1
        n_periods = len(self._values) - 1
        years = n_periods / self.periods_per_year
        if end <= 0:
            annualized_return = -1.0
        else:
            # log/exp with a clamp: short intraday curves can overflow a float power
            growth = math.log(end / start) / years
            annualized_return = math.exp(min(growth, _MAX_LOG_GROWTH)) - 1

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "total_trades": len(self.trades),
            "years": years,
        }

    def _excess_returns(self) -> np.ndarray:
        return np.asarray(self.returns.to_numpy(), dtype=float) - (
            self.risk_free_rate / self.periods_per_year
        )

    def _downside_deviation(self) -> float:
        downside = np.minimum(self._excess_returns(), 0.0)
        return float(np.sqrt(np.mean(downside**2))) if len(downside) else 0.0

    def _volatility(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        vol = float(self.returns.std(ddof=1))
        return vol if math.isfinite(vol) else 0.0

    def _calculate_risk_metrics(self) -> dict:
        """Calculate volatility and downside deviation (per period and annualized)."""
        volatility = self._volatility()
        downside_deviation = self._downside_deviation() if len(self.returns) >= 2 else 0.0
        scale = math.sqrt(self.periods_per_year)
        return {
            "volatility": volatility,
            "annualized_volatility": volatility * scale,
            "downside_deviation": downside_deviation,
            "annualized_downside_deviation": downside_deviation * scale,
        }

    def _calculate_trade_stats(self) -> dict:
        """Calculate trade-level statistics via the shared helper."""
        stats: dict = dict(compute_trade_stats(self.trades))
        stats.pop("total_trades", None)
        return stats

    def _calculate_drawdown_metrics(self) -> dict:
        """Calculate drawdown statistics (including a trailing, unrecovered drawdown)."""
        if len(self._values) < 2:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "avg_drawdown": 0.0,
                "drawdown_periods": 0,
            }

        equity = pd.Series(self._values)
        running_max = equity.cummax()
        drawdown = ((equity - running_max) / running_max.where(running_max > 0)).fillna(0.0)

        drawdown_periods: list[int] = []
        current_duration = 0
        for in_dd in drawdown < 0:
            if in_dd:
                current_duration += 1
            elif current_duration > 0:
                drawdown_periods.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            # Trailing drawdown that has not recovered by the end of the curve
            drawdown_periods.append(current_duration)

        negative = drawdown[drawdown < 0]
        return {
            "max_drawdown": float(drawdown.min()),
            "max_drawdown_duration": max(drawdown_periods) if drawdown_periods else 0,
            "avg_drawdown": float(negative.mean()) if len(negative) else 0.0,
            "drawdown_periods": len(drawdown_periods),
        }

    def _calculate_ratio_metrics(self, annual_return: float, max_dd: float) -> dict:
        """Calculate Sharpe, Sortino and Calmar ratios."""
        scale = math.sqrt(self.periods_per_year)
        sharpe = 0.0
        sortino = 0.0

        if len(self.returns) >= 2:
            mean_excess = float(np.mean(self._excess_returns()))
            vol = self._volatility()
            if vol > _EPS:
                sharpe = mean_excess / vol * scale
            downside_dev = self._downside_deviation()
            if downside_dev > _EPS:
                sortino = mean_excess / downside_dev * scale

        calmar = annual_return / abs(max_dd) if abs(max_dd) > _EPS else 0.0

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
        }

    def _calculate_kelly_parameters(self, trade_stats: dict) -> dict:
        """Calculate Kelly Criterion parameters from trade statistics."""
        win_rate = trade_stats.get("win_rate", 0.0)
        avg_win = trade_stats.get("avg_win", 0.0)
        avg_loss = trade_stats.get("avg_loss", 0.0)

        # Kelly Criterion: f* = (bp - q) / b
        # where b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate
        if avg_loss > 0 and avg_win > 0:
            b = avg_win / avg_loss
            kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        elif avg_loss > 0:
            kelly_fraction = -1.0  # Only losers: bet nothing (negative edge)
        else:
            kelly_fraction = 0.0

        return {
            "kelly_fraction": kelly_fraction,
            "half_kelly": kelly_fraction * 0.5,
            "quarter_kelly": kelly_fraction * 0.25,
            "kelly_win_loss_ratio": avg_win / avg_loss if avg_loss > 0 else 0.0,
        }

    def get_summary(self) -> dict:
        """Get summary of all metrics."""
        return self.metrics

    def print_summary(self) -> None:
        """Print formatted summary report."""
        print("\n" + "=" * 80)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 80)

        print("\n📈 RETURNS:")
        print(f"   Total Return:       {self.metrics['total_return']:>10.2%}")
        print(f"   Annualized Return:  {self.metrics['annualized_return']:>10.2%}")
        print(f"   Years:              {self.metrics['years']:>10.2f}")

        print("\n📊 RISK METRICS:")
        print(f"   Volatility (Ann.):  {self.metrics['annualized_volatility']:>10.2%}")
        print(f"   Max Drawdown:       {self.metrics['max_drawdown']:>10.2%}")
        print(f"   Avg Drawdown:       {self.metrics['avg_drawdown']:>10.2%}")

        print("\n📉 PERFORMANCE RATIOS:")
        print(f"   Sharpe Ratio:       {self.metrics['sharpe_ratio']:>10.2f}")
        print(f"   Sortino Ratio:      {self.metrics['sortino_ratio']:>10.2f}")
        print(f"   Calmar Ratio:       {self.metrics['calmar_ratio']:>10.2f}")

        print("\n💰 TRADE STATISTICS:")
        print(f"   Total Trades:       {self.metrics['total_trades']:>10}")
        print(f"   Winning Trades:     {self.metrics['winning_trades']:>10}")
        print(f"   Losing Trades:      {self.metrics['losing_trades']:>10}")
        print(f"   Win Rate:           {self.metrics['win_rate']:>10.2%}")
        print(f"   Profit Factor:      {self.metrics['profit_factor']:>10.2f}")
        print(f"   Avg Win:            ${self.metrics['avg_win']:>9.2f}")
        print(f"   Avg Loss:           ${self.metrics['avg_loss']:>9.2f}")
        print(f"   Avg Trade:          ${self.metrics['avg_trade']:>9.2f}")
        print(f"   Expectancy:         ${self.metrics['expectancy']:>9.2f}")

        print("\n🎯 KELLY CRITERION PARAMETERS:")
        print(f"   Full Kelly:         {self.metrics['kelly_fraction']:>10.2%}")
        print(f"   Half Kelly:         {self.metrics['half_kelly']:>10.2%}")
        print(f"   Quarter Kelly:      {self.metrics['quarter_kelly']:>10.2%}")
        print(f"   Win/Loss Ratio:     {self.metrics['kelly_win_loss_ratio']:>10.2f}")

        print("\n" + "=" * 80)

    def is_profitable(self, min_sharpe: float = 0.5, min_trades: int = 30) -> bool:
        """
        Determine if strategy is profitable enough for deployment.

        Args:
            min_sharpe: Minimum Sharpe ratio (default: 0.5)
            min_trades: Minimum number of trades for statistical significance

        Returns:
            True if strategy meets profitability criteria
        """
        return bool(
            self.metrics["total_trades"] >= min_trades
            and self.metrics["sharpe_ratio"] >= min_sharpe
            and self.metrics["total_return"] > 0
            and self.metrics["win_rate"] > 0.4  # At least 40% win rate
        )
