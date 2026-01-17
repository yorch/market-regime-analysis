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

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from backtest results.

    This class computes all statistics needed for strategy evaluation
    and Kelly Criterion parameter estimation.
    """

    def __init__(self, trades: list[dict], equity_curve: pd.Series, risk_free_rate: float = 0.02):
        """
        Initialize performance calculator.

        Args:
            trades: List of trade dictionaries with:
                - entry_date, exit_date
                - entry_price, exit_price
                - shares, direction ('LONG' or 'SHORT')
                - pnl (after costs)
                - return_pct
            equity_curve: Time series of portfolio value
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.trades = trades
        self.equity_curve = equity_curve
        self.risk_free_rate = risk_free_rate

        # Calculate metrics
        self.metrics = self._calculate_all_metrics()

    def _calculate_all_metrics(self) -> dict:
        """Calculate comprehensive performance metrics."""
        metrics = {}

        # Basic statistics
        metrics.update(self._calculate_basic_stats())

        # Risk metrics
        metrics.update(self._calculate_risk_metrics())

        # Trade statistics
        metrics.update(self._calculate_trade_stats())

        # Drawdown analysis
        metrics.update(self._calculate_drawdown_metrics())

        # Ratio metrics
        metrics.update(self._calculate_ratio_metrics())

        # Kelly Criterion parameters
        metrics.update(self._calculate_kelly_parameters())

        return metrics

    def _calculate_basic_stats(self) -> dict:
        """Calculate basic return statistics."""
        if len(self.equity_curve) < 2:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "total_trades": 0,
            }

        total_return = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        years = len(self.equity_curve) / 252  # Assuming daily data
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "total_trades": len(self.trades),
            "years": years,
        }

    def _calculate_risk_metrics(self) -> dict:
        """Calculate volatility and risk metrics."""
        if len(self.equity_curve) < 2:
            return {
                "volatility": 0.0,
                "annualized_volatility": 0.0,
            }

        returns = self.equity_curve.pct_change().dropna()
        volatility = returns.std()
        annualized_volatility = volatility * np.sqrt(252)

        # Downside deviation (for Sortino)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.0
        annualized_downside_dev = downside_deviation * np.sqrt(252)

        return {
            "volatility": volatility,
            "annualized_volatility": annualized_volatility,
            "downside_deviation": downside_deviation,
            "annualized_downside_deviation": annualized_downside_dev,
        }

    def _calculate_trade_stats(self) -> dict:
        """Calculate trade-level statistics."""
        if not self.trades:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_trade": 0.0,
                "winning_trades": 0,
                "losing_trades": 0,
            }

        # Separate winning and losing trades
        trade_pnls = [t["pnl"] for t in self.trades]
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]

        # Win rate
        win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0.0

        # Average win/loss
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0

        # Profit factor
        total_wins = sum(winning_trades) if winning_trades else 0.0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0.0
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        else:
            # No losing trades - use large finite value or NaN for undefined
            profit_factor = 999.0 if total_wins > 0 else 0.0

        # Average trade
        avg_trade = np.mean(trade_pnls) if trade_pnls else 0.0

        # Expectancy (average P&L per trade)
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade": avg_trade,
            "expectancy": expectancy,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "total_wins": total_wins,
            "total_losses": total_losses,
        }

    def _calculate_drawdown_metrics(self) -> dict:
        """Calculate drawdown statistics."""
        if len(self.equity_curve) < 2:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "avg_drawdown": 0.0,
            }

        # Calculate drawdown series
        running_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Drawdown duration
        is_in_drawdown = drawdown < 0
        drawdown_periods = []
        current_duration = 0

        for in_dd in is_in_drawdown:
            if in_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                current_duration = 0

        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown = drawdown[drawdown < 0].mean() if any(drawdown < 0) else 0.0

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "avg_drawdown": avg_drawdown,
            "drawdown_periods": len(drawdown_periods),
        }

    def _calculate_ratio_metrics(self) -> dict:
        """Calculate performance ratios."""
        annual_return = (
            self.metrics.get("annualized_return", 0.0) if hasattr(self, "metrics") else 0.0
        )
        annual_vol = (
            self.metrics.get("annualized_volatility", 0.0) if hasattr(self, "metrics") else 0.0
        )
        downside_dev = (
            self.metrics.get("annualized_downside_deviation", 0.0)
            if hasattr(self, "metrics")
            else 0.0
        )
        max_dd = self.metrics.get("max_drawdown", -0.01) if hasattr(self, "metrics") else -0.01

        # Need to get from already calculated metrics
        if not hasattr(self, "metrics"):
            # First pass - calculate from scratch
            basic = self._calculate_basic_stats()
            risk = self._calculate_risk_metrics()
            dd = self._calculate_drawdown_metrics()
            annual_return = basic["annualized_return"]
            annual_vol = risk["annualized_volatility"]
            downside_dev = risk["annualized_downside_deviation"]
            max_dd = dd["max_drawdown"]

        # Sharpe Ratio
        sharpe = ((annual_return - self.risk_free_rate) / annual_vol) if annual_vol > 0 else 0.0

        # Sortino Ratio
        sortino = (
            ((annual_return - self.risk_free_rate) / downside_dev) if downside_dev > 0 else 0.0
        )

        # Calmar Ratio (return / max drawdown)
        calmar = abs(annual_return / max_dd) if max_dd < 0 else 0.0

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
        }

    def _calculate_kelly_parameters(self) -> dict:
        """
        Calculate Kelly Criterion parameters from trade statistics.

        These are the critical parameters missing from the original system!
        """
        if not hasattr(self, "metrics"):
            trade_stats = self._calculate_trade_stats()
        else:
            trade_stats = {
                k: v
                for k, v in self.metrics.items()
                if k in ["win_rate", "avg_win", "avg_loss", "profit_factor"]
            }

        win_rate = trade_stats.get("win_rate", 0.0)
        avg_win = trade_stats.get("avg_win", 0.0)
        avg_loss = trade_stats.get("avg_loss", 0.0)

        # Kelly Criterion: f* = (bp - q) / b
        # where b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate
        if avg_loss > 0:
            b = avg_win / avg_loss  # Win/loss ratio
            p = win_rate
            q = 1 - win_rate
            kelly_fraction = (b * p - q) / b if b > 0 else 0.0
        else:
            kelly_fraction = 0.0

        # Fractional Kelly (safer)
        half_kelly = kelly_fraction * 0.5
        quarter_kelly = kelly_fraction * 0.25

        return {
            "kelly_fraction": kelly_fraction,
            "half_kelly": half_kelly,
            "quarter_kelly": quarter_kelly,
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
        return (
            self.metrics["total_trades"] >= min_trades
            and self.metrics["sharpe_ratio"] >= min_sharpe
            and self.metrics["total_return"] > 0
            and self.metrics["win_rate"] > 0.4  # At least 40% win rate
        )
