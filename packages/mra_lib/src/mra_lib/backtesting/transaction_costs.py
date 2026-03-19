"""
Transaction cost modeling for realistic backtesting.

Models bid-ask spreads, commissions, slippage, and market impact
to ensure backtest results reflect real trading costs.
"""

import numpy as np


class TransactionCostModel:
    """
    Comprehensive transaction cost model.

    Includes:
    - Bid-ask spread costs
    - Commission costs (flat + per-share)
    - Slippage (market impact)
    - Configurable by asset class

    All costs reduce returns - critical for realistic performance estimation.
    """

    def __init__(
        self,
        spread_bps: float = 5.0,  # Bid-ask spread in basis points
        commission_per_share: float = 0.005,  # $0.005 per share
        commission_min: float = 1.0,  # Minimum $1 per trade
        slippage_bps: float = 2.0,  # Slippage in basis points
        market_impact_coeff: float = 0.1,  # Market impact coefficient
    ) -> None:
        """
        Initialize transaction cost model.

        Args:
            spread_bps: Bid-ask spread in basis points (default: 5 bps = 0.05%)
            commission_per_share: Commission per share (default: $0.005)
            commission_min: Minimum commission per trade (default: $1)
            slippage_bps: Slippage in basis points (default: 2 bps)
            market_impact_coeff: Market impact coefficient for large trades
        """
        self.spread_bps = spread_bps
        self.commission_per_share = commission_per_share
        self.commission_min = commission_min
        self.slippage_bps = slippage_bps
        self.market_impact_coeff = market_impact_coeff

    def calculate_total_cost(
        self,
        price: float,
        shares: float,
        direction: str,  # 'BUY' or 'SELL'
        avg_volume: float | None = None,
    ) -> dict[str, float]:
        """
        Calculate total transaction costs for a trade.

        Args:
            price: Execution price
            shares: Number of shares (absolute value)
            direction: 'BUY' or 'SELL'
            avg_volume: Average daily volume (for market impact)

        Returns:
            Dictionary with cost breakdown:
            - spread_cost: Bid-ask spread cost
            - commission: Commission cost
            - slippage: Slippage cost
            - market_impact: Market impact cost (if volume provided)
            - total_cost: Total transaction cost in dollars
            - total_cost_bps: Total cost in basis points
        """
        shares = abs(shares)  # Ensure positive
        notional_value = price * shares

        # 1. Bid-ask spread cost
        spread_cost = notional_value * (self.spread_bps / 10000)

        # 2. Commission cost
        commission = max(self.commission_min, shares * self.commission_per_share)

        # 3. Slippage cost
        slippage_cost = notional_value * (self.slippage_bps / 10000)

        # 4. Market impact cost (if volume data available)
        market_impact_cost = 0.0
        if avg_volume is not None and avg_volume > 0:
            # Impact proportional to sqrt(shares / avg_volume)
            participation_rate = shares / avg_volume
            market_impact_cost = (
                notional_value * self.market_impact_coeff * np.sqrt(participation_rate)
            )

        # Total cost
        total_cost = spread_cost + commission + slippage_cost + market_impact_cost

        # Express as basis points for comparison
        total_cost_bps = (total_cost / notional_value) * 10000 if notional_value > 0 else 0

        return {
            "spread_cost": spread_cost,
            "commission": commission,
            "slippage": slippage_cost,
            "market_impact": market_impact_cost,
            "total_cost": total_cost,
            "total_cost_bps": total_cost_bps,
            "notional_value": notional_value,
        }

    def calculate_roundtrip_cost(
        self, price: float, shares: float, avg_volume: float | None = None
    ) -> float:
        """
        Calculate round-trip (buy + sell) transaction cost.

        Args:
            price: Execution price
            shares: Number of shares
            avg_volume: Average daily volume

        Returns:
            Total round-trip cost in dollars
        """
        buy_costs = self.calculate_total_cost(price, shares, "BUY", avg_volume)
        sell_costs = self.calculate_total_cost(price, shares, "SELL", avg_volume)

        return buy_costs["total_cost"] + sell_costs["total_cost"]

    def get_minimum_profit_threshold(
        self, price: float, shares: float, avg_volume: float | None = None
    ) -> float:
        """
        Calculate minimum profit needed to overcome transaction costs.

        Trading rule: Only enter if expected profit > 2x transaction costs

        Args:
            price: Entry price
            shares: Position size
            avg_volume: Average daily volume

        Returns:
            Minimum price move (in dollars) needed for break-even after costs
        """
        roundtrip_cost = self.calculate_roundtrip_cost(price, shares, avg_volume)
        # Return as percentage of notional value
        notional = price * shares
        return roundtrip_cost / notional if notional > 0 else 0.0

    def apply_costs_to_returns(
        self,
        entry_price: float,
        exit_price: float,
        shares: float,
        direction: str,  # 'LONG' or 'SHORT'
        avg_volume: float | None = None,
    ) -> dict[str, float]:
        """
        Calculate P&L after transaction costs.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            shares: Number of shares
            direction: 'LONG' or 'SHORT'
            avg_volume: Average daily volume

        Returns:
            Dictionary with:
            - gross_pnl: P&L before costs
            - entry_costs: Entry transaction costs
            - exit_costs: Exit transaction costs
            - net_pnl: P&L after costs
            - return_pct: Net return percentage
        """
        shares = abs(shares)

        # Calculate gross P&L
        if direction == "LONG":
            gross_pnl = (exit_price - entry_price) * shares
        else:  # SHORT
            gross_pnl = (entry_price - exit_price) * shares

        # Calculate transaction costs
        entry_costs = self.calculate_total_cost(
            entry_price, shares, "BUY" if direction == "LONG" else "SELL", avg_volume
        )
        exit_costs = self.calculate_total_cost(
            exit_price, shares, "SELL" if direction == "LONG" else "BUY", avg_volume
        )

        total_costs = entry_costs["total_cost"] + exit_costs["total_cost"]
        net_pnl = gross_pnl - total_costs

        # Calculate return percentage
        notional = entry_price * shares
        return_pct = (net_pnl / notional * 100) if notional > 0 else 0.0

        return {
            "gross_pnl": gross_pnl,
            "entry_costs": entry_costs["total_cost"],
            "exit_costs": exit_costs["total_cost"],
            "total_costs": total_costs,
            "net_pnl": net_pnl,
            "return_pct": return_pct,
            "notional_value": notional,
        }


# Preset cost models for different asset classes
class EquityCostModel(TransactionCostModel):
    """US equity transaction costs (low-cost broker)."""

    def __init__(self):
        super().__init__(
            spread_bps=5.0,  # 0.05% spread
            commission_per_share=0.005,  # Half cent per share
            commission_min=1.0,  # $1 minimum
            slippage_bps=2.0,  # 0.02% slippage
            market_impact_coeff=0.1,
        )


class FuturesCostModel(TransactionCostModel):
    """Futures transaction costs."""

    def __init__(self):
        super().__init__(
            spread_bps=2.0,  # Tighter spreads
            commission_per_share=2.5,  # Per contract
            commission_min=2.5,
            slippage_bps=1.0,  # Lower slippage
            market_impact_coeff=0.05,
        )


class HighFrequencyCostModel(TransactionCostModel):
    """High-frequency trading with tight spreads."""

    def __init__(self):
        super().__init__(
            spread_bps=1.0,  # Very tight
            commission_per_share=0.001,  # Maker-taker pricing
            commission_min=0.0,
            slippage_bps=0.5,
            market_impact_coeff=0.05,
        )


class RetailCostModel(TransactionCostModel):
    """Retail trading with higher costs."""

    def __init__(self):
        super().__init__(
            spread_bps=10.0,  # Wider spreads
            commission_per_share=0.0,  # Zero commission brokers
            commission_min=0.0,
            slippage_bps=5.0,  # Higher slippage
            market_impact_coeff=0.2,
        )
