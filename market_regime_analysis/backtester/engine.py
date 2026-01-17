"""
Backtesting engine for strategy validation.

Core backtesting functionality for testing regime-based trading strategies
with realistic execution, transaction costs, and performance measurement.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..enums import MarketRegime, TradingStrategy
from .metrics import PerformanceMetrics
from .transaction_costs import EquityCostModel, TransactionCostModel


@dataclass
class Trade:
    """Represents a completed trade."""

    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: float
    direction: str  # 'LONG' or 'SHORT'
    entry_regime: str
    exit_regime: str
    gross_pnl: float
    net_pnl: float
    return_pct: float
    entry_costs: float
    exit_costs: float
    holding_days: int


class BacktestEngine:
    """
    Backtest regime-based trading strategies.

    Simulates historical trading with:
    - Regime detection
    - Position sizing
    - Transaction costs
    - Stop-loss/take-profit
    - Performance tracking
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        cost_model: TransactionCostModel | None = None,
        max_position_size: float = 0.20,  # 20% max per position
        stop_loss_pct: float | None = 0.10,  # 10% stop loss
        take_profit_pct: float | None = None,  # No take profit by default
    ) -> None:
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            cost_model: Transaction cost model (default: EquityCostModel)
            max_position_size: Maximum position size as fraction of capital
            stop_loss_pct: Stop loss percentage (None to disable)
            take_profit_pct: Take profit percentage (None to disable)
        """
        self.initial_capital = initial_capital
        self.cost_model = cost_model or EquityCostModel()
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Backtest state
        self.capital = initial_capital
        self.position: dict | None = None  # Current open position
        self.trades: list[dict] = []
        self.equity_curve: list[float] = [initial_capital]
        self.dates: list[datetime] = []

    def run_regime_strategy(
        self,
        df: pd.DataFrame,
        regimes: pd.Series,  # Series of MarketRegime enum values
        strategies: pd.Series,  # Series of TradingStrategy enum values
        position_sizes: pd.Series,  # Series of position size multipliers
    ) -> dict:
        """
        Run backtest for regime-based strategy.

        Args:
            df: OHLC price data
            regimes: Market regime for each date
            strategies: Recommended strategy for each date
            position_sizes: Position size multiplier for each date

        Returns:
            Dictionary with backtest results
        """
        # Reset state
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.dates = [df.index[0]]

        # Iterate through each day
        for i in range(len(df)):
            date = df.index[i]
            price = df["Close"].iloc[i]
            regime = regimes.iloc[i] if i < len(regimes) else MarketRegime.UNKNOWN
            strategy = strategies.iloc[i] if i < len(strategies) else TradingStrategy.AVOID
            position_mult = position_sizes.iloc[i] if i < len(position_sizes) else 0.0

            # Update existing position
            if self.position is not None:
                # Check stop-loss/take-profit
                self._check_exit_conditions(date, price, df["High"].iloc[i], df["Low"].iloc[i])

                # Check regime change exit
                if self._should_exit_regime(regime, strategy):
                    self._close_position(date, price, regime.value if isinstance(regime, MarketRegime) else str(regime))

            # Enter new position if no current position
            if self.position is None:
                if self._should_enter_position(strategy):
                    direction = self._get_direction_from_strategy(strategy)
                    if direction:
                        size = self._calculate_position_size(price, position_mult)
                        self._open_position(date, price, size, direction, regime.value if isinstance(regime, MarketRegime) else str(regime))

            # Update equity curve
            current_equity = self._calculate_current_equity(price)
            self.equity_curve.append(current_equity)
            self.dates.append(date)

        # Close any remaining position
        if self.position is not None:
            final_price = df["Close"].iloc[-1]
            final_date = df.index[-1]
            self._close_position(final_date, final_price, "END")

        # Calculate performance metrics
        equity_series = pd.Series(self.equity_curve, index=self.dates)
        performance = PerformanceMetrics(self.trades, equity_series)

        return {
            "trades": self.trades,
            "equity_curve": equity_series,
            "performance": performance,
            "final_capital": self.capital,
            "total_return": (self.capital / self.initial_capital - 1),
        }

    def _should_enter_position(self, strategy: TradingStrategy) -> bool:
        """Determine if we should enter a position based on strategy."""
        # Only enter on directional strategies
        return strategy in [
            TradingStrategy.TREND_FOLLOWING,
            TradingStrategy.MOMENTUM,
            TradingStrategy.MEAN_REVERSION,
        ]

    def _should_exit_regime(self, regime: MarketRegime, strategy: TradingStrategy) -> bool:
        """Check if regime change signals exit."""
        # Exit if strategy changes to defensive or avoid
        if strategy in [TradingStrategy.DEFENSIVE, TradingStrategy.AVOID]:
            return True

        # Exit if regime is unknown or high volatility
        if regime in [MarketRegime.UNKNOWN, MarketRegime.HIGH_VOLATILITY]:
            return True

        return False

    def _get_direction_from_strategy(self, strategy: TradingStrategy) -> str | None:
        """Get position direction from strategy."""
        if strategy == TradingStrategy.TREND_FOLLOWING:
            return "LONG"  # Simplified: always long in trending
        elif strategy == TradingStrategy.MOMENTUM:
            return "LONG"
        elif strategy == TradingStrategy.MEAN_REVERSION:
            # Could be long or short depending on z-score
            # Simplified: long
            return "LONG"
        return None

    def _calculate_position_size(self, price: float, position_mult: float) -> float:
        """
        Calculate position size in shares.

        Args:
            price: Current price
            position_mult: Regime-based multiplier

        Returns:
            Number of shares to buy
        """
        # Base position size (fraction of capital)
        base_fraction = 0.10  # 10% base position

        # Apply regime multiplier
        target_fraction = base_fraction * position_mult

        # Cap at maximum
        target_fraction = min(target_fraction, self.max_position_size)

        # Calculate shares
        target_dollars = self.capital * target_fraction
        shares = int(target_dollars / price)

        return shares

    def _open_position(
        self, date: datetime, price: float, shares: float, direction: str, regime: str
    ) -> None:
        """Open a new position."""
        if shares <= 0:
            return

        # Calculate entry costs
        costs = self.cost_model.calculate_total_cost(price, shares, "BUY" if direction == "LONG" else "SELL")

        # Deduct costs from capital
        self.capital -= costs["total_cost"]

        # Create position
        self.position = {
            "entry_date": date,
            "entry_price": price,
            "shares": shares,
            "direction": direction,
            "entry_regime": regime,
            "entry_costs": costs["total_cost"],
        }

    def _close_position(self, date: datetime, price: float, regime: str) -> None:
        """Close current position."""
        if self.position is None:
            return

        shares = self.position["shares"]
        direction = self.position["direction"]

        # Calculate exit costs
        costs = self.cost_model.calculate_total_cost(price, shares, "SELL" if direction == "LONG" else "BUY")

        # Calculate P&L
        if direction == "LONG":
            gross_pnl = (price - self.position["entry_price"]) * shares
        else:
            gross_pnl = (self.position["entry_price"] - price) * shares

        total_costs = self.position["entry_costs"] + costs["total_cost"]
        net_pnl = gross_pnl - total_costs

        # Update capital
        notional = self.position["entry_price"] * shares
        self.capital += notional + net_pnl

        # Calculate return percentage
        return_pct = (net_pnl / notional * 100) if notional > 0 else 0.0

        # Record trade
        holding_days = (date - self.position["entry_date"]).days

        trade = {
            "entry_date": self.position["entry_date"],
            "exit_date": date,
            "entry_price": self.position["entry_price"],
            "exit_price": price,
            "shares": shares,
            "direction": direction,
            "entry_regime": self.position["entry_regime"],
            "exit_regime": regime,
            "gross_pnl": gross_pnl,
            "pnl": net_pnl,
            "return_pct": return_pct,
            "entry_costs": self.position["entry_costs"],
            "exit_costs": costs["total_cost"],
            "holding_days": holding_days,
        }

        self.trades.append(trade)
        self.position = None

    def _check_exit_conditions(
        self, date: datetime, close: float, high: float, low: float
    ) -> None:
        """Check if stop-loss or take-profit hit."""
        if self.position is None:
            return

        entry_price = self.position["entry_price"]
        direction = self.position["direction"]

        # Calculate current P&L percentage
        if direction == "LONG":
            current_return = (close - entry_price) / entry_price
        else:
            current_return = (entry_price - close) / entry_price

        # Check stop-loss
        if self.stop_loss_pct and current_return < -self.stop_loss_pct:
            self._close_position(date, close, "STOP_LOSS")
            return

        # Check take-profit
        if self.take_profit_pct and current_return > self.take_profit_pct:
            self._close_position(date, close, "TAKE_PROFIT")
            return

    def _calculate_current_equity(self, current_price: float) -> float:
        """Calculate current portfolio equity."""
        equity = self.capital

        # Add unrealized P&L from open position
        if self.position is not None:
            shares = self.position["shares"]
            entry_price = self.position["entry_price"]
            direction = self.position["direction"]

            if direction == "LONG":
                unrealized_pnl = (current_price - entry_price) * shares
            else:
                unrealized_pnl = (entry_price - current_price) * shares

            notional = entry_price * shares
            equity += notional + unrealized_pnl

        return equity

    def print_results(self, results: dict) -> None:
        """Print backtest results."""
        print("\n" + "=" * 100)
        print("BACKTEST RESULTS")
        print("=" * 100)

        print(f"\n💰 CAPITAL:")
        print(f"   Initial Capital:    ${self.initial_capital:>12,.2f}")
        print(f"   Final Capital:      ${results['final_capital']:>12,.2f}")
        print(f"   Total Return:       {results['total_return']:>13.2%}")

        print(f"\n📊 TRADES:")
        print(f"   Total Trades:       {len(results['trades']):>12}")

        if results["trades"]:
            winning = [t for t in results["trades"] if t["pnl"] > 0]
            losing = [t for t in results["trades"] if t["pnl"] < 0]
            print(f"   Winning Trades:     {len(winning):>12}")
            print(f"   Losing Trades:      {len(losing):>12}")

        # Delegate to PerformanceMetrics for detailed stats
        if "performance" in results:
            results["performance"].print_summary()
