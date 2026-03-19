"""
Simons Risk Calculator for professional risk management.

This module implements Renaissance Technologies' approach to position sizing
and risk management, including Kelly Criterion optimization and regime-adjusted
position sizing.
"""

from dataclasses import dataclass

from .enums import MarketRegime


@dataclass
class PositionRecord:
    """Tracks a single open position for portfolio-level limit enforcement."""

    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    notional: float  # position notional value
    sector: str = ""  # optional sector/group tag


class PortfolioPositionLimits:
    """
    Cross-asset position limit enforcer.

    Tracks open positions across multiple symbols and enforces:
    - Maximum total portfolio exposure (gross notional / capital)
    - Maximum per-asset exposure
    - Maximum number of concurrent positions
    - Maximum net directional exposure (long - short)
    - Maximum sector/group concentration

    All limits are expressed as fractions of portfolio capital.
    """

    def __init__(
        self,
        capital: float,
        max_total_exposure: float = 1.0,
        max_per_asset_exposure: float = 0.20,
        max_positions: int = 20,
        max_net_exposure: float = 0.60,
        max_sector_exposure: float = 0.40,
        max_correlated_exposure: float = 0.50,
    ) -> None:
        """
        Initialize portfolio position limits.

        Args:
            capital: Current portfolio capital
            max_total_exposure: Max gross exposure as fraction of capital (default 1.0 = 100%)
            max_per_asset_exposure: Max single-asset exposure (default 20%)
            max_positions: Max number of concurrent open positions
            max_net_exposure: Max net directional exposure (|long - short| / capital)
            max_sector_exposure: Max exposure to a single sector/group
            max_correlated_exposure: Max combined exposure for correlated assets
        """
        self.capital = capital
        self.max_total_exposure = max_total_exposure
        self.max_per_asset_exposure = max_per_asset_exposure
        self.max_positions = max_positions
        self.max_net_exposure = max_net_exposure
        self.max_sector_exposure = max_sector_exposure
        self.max_correlated_exposure = max_correlated_exposure

        self.positions: dict[str, PositionRecord] = {}

    def update_capital(self, capital: float) -> None:
        """Update current portfolio capital."""
        self.capital = capital

    def add_position(self, position: PositionRecord) -> None:
        """Register an open position."""
        self.positions[position.symbol] = position

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        self.positions.pop(symbol, None)

    def get_gross_exposure(self) -> float:
        """Total absolute notional across all positions."""
        return sum(abs(p.notional) for p in self.positions.values())

    def get_net_exposure(self) -> float:
        """Net directional exposure (long notional - short notional)."""
        net = 0.0
        for p in self.positions.values():
            net += p.notional if p.direction == "LONG" else -p.notional
        return net

    def get_long_exposure(self) -> float:
        """Total long notional."""
        return sum(p.notional for p in self.positions.values() if p.direction == "LONG")

    def get_short_exposure(self) -> float:
        """Total short notional."""
        return sum(p.notional for p in self.positions.values() if p.direction == "SHORT")

    def get_sector_exposure(self, sector: str) -> float:
        """Total notional for a given sector."""
        return sum(abs(p.notional) for p in self.positions.values() if p.sector == sector)

    def get_asset_exposure(self, symbol: str) -> float:
        """Current notional for a specific asset."""
        p = self.positions.get(symbol)
        return abs(p.notional) if p else 0.0

    def check_limits(
        self,
        symbol: str,
        direction: str,
        proposed_notional: float,
        sector: str = "",
    ) -> dict:
        """
        Check whether a proposed trade would violate any position limits.

        Args:
            symbol: Asset symbol
            direction: 'LONG' or 'SHORT'
            proposed_notional: Notional value of proposed trade
            sector: Optional sector/group tag

        Returns:
            Dictionary with:
                - allowed: bool - whether the trade is permitted
                - max_allowed_notional: float - largest notional that would pass all limits
                - violations: list[str] - descriptions of any limit breaches
        """
        violations: list[str] = []
        cap = self.capital if self.capital > 0 else 1.0  # avoid division by zero

        proposed_abs = abs(proposed_notional)

        # Existing exposures (exclude current position in same symbol if replacing)
        current_gross = self.get_gross_exposure() - self.get_asset_exposure(symbol)
        current_net = self.get_net_exposure()
        if symbol in self.positions:
            old = self.positions[symbol]
            current_net -= old.notional if old.direction == "LONG" else -old.notional

        # 1. Max positions count
        new_count = len(self.positions) + (0 if symbol in self.positions else 1)
        if new_count > self.max_positions:
            violations.append(f"Max positions exceeded: {new_count} > {self.max_positions}")

        # 2. Per-asset exposure
        if proposed_abs / cap > self.max_per_asset_exposure:
            violations.append(
                f"Per-asset exposure {proposed_abs / cap:.1%} > "
                f"limit {self.max_per_asset_exposure:.1%}"
            )

        # 3. Total gross exposure
        new_gross = current_gross + proposed_abs
        if new_gross / cap > self.max_total_exposure:
            violations.append(
                f"Total exposure {new_gross / cap:.1%} > limit {self.max_total_exposure:.1%}"
            )

        # 4. Net directional exposure
        signed = proposed_abs if direction == "LONG" else -proposed_abs
        new_net = current_net + signed
        if abs(new_net) / cap > self.max_net_exposure:
            violations.append(
                f"Net exposure {abs(new_net) / cap:.1%} > limit {self.max_net_exposure:.1%}"
            )

        # 5. Sector concentration
        if sector:
            current_sector = self.get_sector_exposure(sector)
            # Remove same symbol's old sector contribution
            if symbol in self.positions and self.positions[symbol].sector == sector:
                current_sector -= self.get_asset_exposure(symbol)
            new_sector = current_sector + proposed_abs
            if new_sector / cap > self.max_sector_exposure:
                violations.append(
                    f"Sector '{sector}' exposure {new_sector / cap:.1%} > "
                    f"limit {self.max_sector_exposure:.1%}"
                )

        # Calculate max allowed notional (the tightest binding constraint)
        max_allowed = proposed_abs
        # Per-asset limit
        max_allowed = min(max_allowed, cap * self.max_per_asset_exposure)
        # Gross exposure headroom
        gross_headroom = cap * self.max_total_exposure - current_gross
        max_allowed = min(max_allowed, max(0.0, gross_headroom))
        # Net exposure headroom
        net_headroom = cap * self.max_net_exposure - abs(current_net)
        max_allowed = min(max_allowed, max(0.0, net_headroom))

        return {
            "allowed": len(violations) == 0,
            "max_allowed_notional": max_allowed,
            "violations": violations,
        }

    def clamp_position_size(
        self,
        symbol: str,
        direction: str,
        desired_notional: float,
        sector: str = "",
    ) -> float:
        """
        Return the largest position size that respects all limits.

        Args:
            symbol: Asset symbol
            direction: 'LONG' or 'SHORT'
            desired_notional: Desired notional value
            sector: Optional sector tag

        Returns:
            Clamped notional value (may be 0 if no room)
        """
        result = self.check_limits(symbol, direction, desired_notional, sector)
        return min(abs(desired_notional), result["max_allowed_notional"])

    def get_portfolio_summary(self) -> dict:
        """Get summary of current portfolio exposure."""
        cap = self.capital if self.capital > 0 else 1.0
        gross = self.get_gross_exposure()
        net = self.get_net_exposure()

        # Sector breakdown
        sectors: dict[str, float] = {}
        for p in self.positions.values():
            if p.sector:
                sectors[p.sector] = sectors.get(p.sector, 0.0) + abs(p.notional)

        return {
            "capital": self.capital,
            "n_positions": len(self.positions),
            "gross_exposure": gross,
            "gross_exposure_pct": gross / cap,
            "net_exposure": net,
            "net_exposure_pct": abs(net) / cap,
            "long_exposure": self.get_long_exposure(),
            "short_exposure": self.get_short_exposure(),
            "sector_exposures": {s: v / cap for s, v in sectors.items()},
            "headroom_gross": max(0.0, cap * self.max_total_exposure - gross),
            "headroom_net": max(0.0, cap * self.max_net_exposure - abs(net)),
        }


class SimonsRiskCalculator:
    """
    Professional risk management following Renaissance approach.

    This class implements sophisticated risk management techniques including
    Kelly Criterion optimization, regime-adjusted position sizing, and
    correlation-based adjustments following Jim Simons' methodology.
    """

    @staticmethod
    def calculate_kelly_optimal_size(
        win_rate: float, avg_win: float, avg_loss: float, confidence: float = 1.0
    ) -> float:
        """
        Calculate Kelly Criterion optimal position size with confidence scaling.

        The Kelly Criterion determines the optimal fraction of capital to risk
        based on the edge and odds of a trading strategy.

        Formula: f* = (bp - q) / b
        Where:
        - b = odds (avg_win / avg_loss)
        - p = probability of winning
        - q = probability of losing (1-p)

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount (positive)
            avg_loss: Average loss amount (positive)
            confidence: Confidence factor to scale down Kelly (0-1)

        Returns:
            Optimal position size as fraction of capital (0-1)

        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        if not (0 <= win_rate <= 1):
            raise ValueError("Win rate must be between 0 and 1")
        if avg_win <= 0:
            raise ValueError("Average win must be positive")
        if avg_loss <= 0:
            raise ValueError("Average loss must be positive")
        if not (0 <= confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")

        # Handle edge cases
        if win_rate == 0:
            return 0.0  # Never bet if win rate is 0
        if win_rate == 1:
            return confidence  # Bet everything if win rate is 100%

        # Calculate Kelly fraction
        b = avg_win / avg_loss  # Odds
        p = win_rate  # Probability of winning
        q = 1 - p  # Probability of losing

        # Kelly formula: f* = (bp - q) / b
        kelly_fraction = (b * p - q) / b

        # Only bet if we have an edge (positive Kelly)
        if kelly_fraction <= 0:
            return 0.0

        # Apply confidence scaling and safety cap
        adjusted_fraction = kelly_fraction * confidence

        # Safety cap at 25% for any single position
        return min(adjusted_fraction, 0.25)

    @staticmethod
    def calculate_regime_adjusted_size(
        base_size: float, regime: MarketRegime, confidence: float, persistence: float
    ) -> float:
        """
        Calculate multi-factor position sizing with regime adjustments.

        This method implements Renaissance Technologies' approach to position
        sizing by incorporating market regime, confidence in regime detection,
        and regime persistence.

        Args:
            base_size: Base position size (fraction of capital)
            regime: Current market regime
            confidence: Confidence in regime classification (0-1)
            persistence: Regime persistence metric (0-1)

        Returns:
            Adjusted position size

        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        if not (0 <= base_size <= 1):
            raise ValueError("Base size must be between 0 and 1")
        if not (0 <= confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        if not (0 <= persistence <= 1):
            raise ValueError("Persistence must be between 0 and 1")

        # Regime multipliers (Renaissance approach)
        regime_multipliers = {
            MarketRegime.BULL_TRENDING: 1.3,
            MarketRegime.BEAR_TRENDING: 0.7,
            MarketRegime.MEAN_REVERTING: 1.2,
            MarketRegime.HIGH_VOLATILITY: 0.4,
            MarketRegime.LOW_VOLATILITY: 1.1,
            MarketRegime.BREAKOUT: 0.9,
            MarketRegime.UNKNOWN: 0.2,
        }

        # Get base regime multiplier
        base_multiplier = regime_multipliers.get(regime, 0.2)

        # Confidence scaling (scale between 0.3 and 1.0)
        confidence_factor = 0.3 + (confidence * 0.7)

        # Persistence adjustment (scale between 0.7 and 1.0)
        persistence_factor = 0.7 + (persistence * 0.3)

        # Combined adjustment
        total_multiplier = base_multiplier * confidence_factor * persistence_factor

        # Calculate final size
        adjusted_size = base_size * total_multiplier

        # Safety caps (1% minimum, 50% maximum)
        return max(0.01, min(0.5, adjusted_size))

    @staticmethod
    def calculate_correlation_adjusted_size(base_size: float, correlation: float) -> float:
        """
        Adjust position size based on correlation with existing positions.

        Higher correlation with existing positions should reduce position size
        to maintain portfolio diversification.

        Args:
            base_size: Base position size
            correlation: Correlation with existing portfolio (-1 to 1)

        Returns:
            Correlation-adjusted position size

        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        if not (0 <= base_size <= 1):
            raise ValueError("Base size must be between 0 and 1")
        if not (-1 <= correlation <= 1):
            raise ValueError("Correlation must be between -1 and 1")

        # Correlation adjustment factor
        # High positive correlation reduces size, negative correlation may increase
        abs_correlation = abs(correlation)

        if abs_correlation < 0.3:
            # Low correlation - no adjustment
            correlation_factor = 1.0
        elif abs_correlation < 0.7:
            # Medium correlation - moderate reduction
            correlation_factor = 1.0 - (abs_correlation - 0.3) * 0.5
        else:
            # High correlation - significant reduction
            correlation_factor = 0.8 - (abs_correlation - 0.7) * 1.0

        # Ensure factor doesn't go below 0.2
        correlation_factor = max(0.2, correlation_factor)

        # Apply adjustment
        adjusted_size = base_size * correlation_factor

        return max(0.01, min(0.5, adjusted_size))

    @staticmethod
    def calculate_volatility_adjusted_size(
        base_size: float,
        current_volatility: float,
        historical_volatility: float,
        vol_target: float = 0.15,
    ) -> float:
        """
        Adjust position size based on volatility conditions.

        This method implements volatility targeting, scaling position sizes
        to maintain consistent risk exposure across different volatility regimes.

        Args:
            base_size: Base position size
            current_volatility: Current asset volatility (annualized)
            historical_volatility: Historical average volatility (annualized)
            vol_target: Target volatility level (default 15%)

        Returns:
            Volatility-adjusted position size

        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        if not (0 <= base_size <= 1):
            raise ValueError("Base size must be between 0 and 1")
        if current_volatility <= 0:
            raise ValueError("Current volatility must be positive")
        if historical_volatility <= 0:
            raise ValueError("Historical volatility must be positive")
        if vol_target <= 0:
            raise ValueError("Volatility target must be positive")

        # Calculate volatility scaling factor
        vol_ratio = current_volatility / historical_volatility
        target_scaling = vol_target / current_volatility

        # Combine both adjustments
        vol_adjustment = target_scaling * (1.0 / vol_ratio)

        # Apply bounds to prevent extreme adjustments
        vol_adjustment = max(0.1, min(3.0, vol_adjustment))

        # Calculate adjusted size
        adjusted_size = base_size * vol_adjustment

        return max(0.01, min(0.5, adjusted_size))

    @staticmethod
    def calculate_comprehensive_position_size(
        base_size: float,
        regime: MarketRegime,
        confidence: float,
        persistence: float,
        correlation: float = 0.0,
        win_rate: float | None = None,
        avg_win: float | None = None,
        avg_loss: float | None = None,
        current_vol: float | None = None,
        historical_vol: float | None = None,
    ) -> dict[str, float]:
        """
        Calculate comprehensive position size using all available factors.

        This method combines all risk management techniques into a single
        comprehensive position sizing calculation.

        Args:
            base_size: Base position size
            regime: Current market regime
            confidence: Regime confidence
            persistence: Regime persistence
            correlation: Portfolio correlation
            win_rate: Strategy win rate (optional)
            avg_win: Average win amount (optional)
            avg_loss: Average loss amount (optional)
            current_vol: Current volatility (optional)
            historical_vol: Historical volatility (optional)

        Returns:
            Dictionary with various position size calculations
        """
        results = {
            "base_size": base_size,
            "regime_adjusted": 0.0,
            "correlation_adjusted": 0.0,
            "kelly_optimal": 0.0,
            "volatility_adjusted": 0.0,
            "final_size": 0.0,
        }

        try:
            # Regime adjustment (always calculated)
            regime_size = SimonsRiskCalculator.calculate_regime_adjusted_size(
                base_size, regime, confidence, persistence
            )
            results["regime_adjusted"] = regime_size

            # Correlation adjustment
            corr_adjusted = SimonsRiskCalculator.calculate_correlation_adjusted_size(
                regime_size, correlation
            )
            results["correlation_adjusted"] = corr_adjusted

            # Kelly criterion (if strategy stats available)
            if all(x is not None for x in [win_rate, avg_win, avg_loss]):
                assert win_rate is not None
                assert avg_win is not None
                assert avg_loss is not None
                kelly_size = SimonsRiskCalculator.calculate_kelly_optimal_size(
                    win_rate, avg_win, avg_loss, confidence
                )
                results["kelly_optimal"] = kelly_size

                # Use Kelly as final size if available and reasonable
                final_size = min(corr_adjusted, kelly_size)
            else:
                final_size = corr_adjusted

            # Volatility adjustment (if volatility data available)
            if current_vol is not None and historical_vol is not None:
                vol_adjusted = SimonsRiskCalculator.calculate_volatility_adjusted_size(
                    final_size, current_vol, historical_vol
                )
                results["volatility_adjusted"] = vol_adjusted
                final_size = vol_adjusted
            else:
                results["volatility_adjusted"] = final_size

            results["final_size"] = final_size

        except Exception as e:
            print(f"Error in comprehensive position sizing: {e!s}")
            results["final_size"] = max(0.01, min(0.1, base_size))

        return results
