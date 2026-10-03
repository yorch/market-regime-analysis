"""
Kelly-based position sizing and risk management.

This module implements position sizing and risk management, including Kelly Criterion optimization and regime-adjusted
position sizing.
"""

from dataclasses import dataclass

from mra_lib.config.enums import MarketRegime
from mra_lib.config.regime_tables import get_regime_multiplier

MIN_POSITION_SIZE = 0.01
"""Smallest non-zero position (fraction of capital). Applied only when size > 0."""

MAX_POSITION_SIZE = 0.50
"""Largest position any single sizing step may return (fraction of capital)."""

MAX_KELLY_FRACTION = 0.25
"""Safety cap on the Kelly fraction for a single position."""


def _bound_position(size: float, upper: float = MAX_POSITION_SIZE) -> float:
    """
    Clamp a position size to ``[MIN_POSITION_SIZE, upper]``, keeping zero at zero.

    A size of zero (or below) means "no edge / do not trade" and is returned as
    ``0.0``; the 1% floor only lifts *positive* sizes that are too small to be
    practical.
    """
    if size <= 0:
        return 0.0
    return max(MIN_POSITION_SIZE, min(upper, size))


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
        """
        self.capital = capital
        self.max_total_exposure = max_total_exposure
        self.max_per_asset_exposure = max_per_asset_exposure
        self.max_positions = max_positions
        self.max_net_exposure = max_net_exposure
        self.max_sector_exposure = max_sector_exposure

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

    def update_position_notional(self, symbol: str, current_price: float, shares: float) -> None:
        """
        Update a position's notional to reflect current market value.

        Should be called each bar so that exposure limits track mark-to-market
        values rather than stale entry notionals.

        Args:
            symbol: Asset symbol
            current_price: Current market price
            shares: Number of shares held
        """
        if symbol in self.positions:
            self.positions[symbol].notional = abs(current_price * shares)

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
        # Net exposure headroom (direction-aware: a trade that moves net exposure
        # toward zero, i.e. a hedge, has more room than one that extends it)
        if direction == "LONG":
            net_headroom = cap * self.max_net_exposure - current_net
        else:
            net_headroom = cap * self.max_net_exposure + current_net
        max_allowed = min(max_allowed, max(0.0, net_headroom))
        # Sector exposure headroom
        if sector:
            current_sector = self.get_sector_exposure(sector)
            if symbol in self.positions and self.positions[symbol].sector == sector:
                current_sector -= self.get_asset_exposure(symbol)
            sector_headroom = cap * self.max_sector_exposure - current_sector
            max_allowed = min(max_allowed, max(0.0, sector_headroom))

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
        return min(abs(desired_notional), float(result["max_allowed_notional"]))

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
    Kelly-based position sizing with regime and correlation adjustments.

    This class implements sophisticated risk management techniques including
    Kelly Criterion optimization, regime-adjusted position sizing, and
    correlation-based adjustments.
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

        # Calculate Kelly fraction (win_rate == 1 gives f* = 1, still capped below)
        b = avg_win / avg_loss  # Odds
        p = win_rate  # Probability of winning
        q = 1 - p  # Probability of losing

        # Kelly formula: f* = (bp - q) / b
        kelly_fraction = (b * p - q) / b

        # Only bet if we have an edge (positive Kelly)
        if kelly_fraction <= 0:
            return 0.0

        # Apply confidence scaling and the per-position safety cap
        return min(kelly_fraction * confidence, MAX_KELLY_FRACTION)

    @staticmethod
    def regime_sizing_factors(
        regime: MarketRegime, confidence: float, persistence: float
    ) -> tuple[float, float, float]:
        """
        Return the factors ``calculate_regime_adjusted_size`` multiplies together.

        Returns:
            ``(regime_multiplier, confidence_factor, persistence_factor)`` where the
            regime multiplier comes from ``config/regime_tables.py`` (UNKNOWN -> 0.0),
            the confidence factor scales between 0.3 and 1.0 and the persistence
            factor between 0.7 and 1.0.
        """
        return (
            get_regime_multiplier(regime),
            0.3 + (confidence * 0.7),
            0.7 + (persistence * 0.3),
        )

    @staticmethod
    def calculate_regime_adjusted_size(
        base_size: float, regime: MarketRegime, confidence: float, persistence: float
    ) -> float:
        """
        Calculate multi-factor position sizing with regime adjustments.

        Scales position size by incorporating market regime, confidence in regime detection,
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

        base_multiplier, confidence_factor, persistence_factor = (
            SimonsRiskCalculator.regime_sizing_factors(regime, confidence, persistence)
        )

        # Combined adjustment
        total_multiplier = base_multiplier * confidence_factor * persistence_factor

        # Calculate final size; 0 stays 0, positive sizes are bounded to [1%, 50%]
        return _bound_position(base_size * total_multiplier)

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

        # Apply adjustment; 0 stays 0, positive sizes are bounded to [1%, 50%]
        return _bound_position(base_size * correlation_factor)

    @staticmethod
    def calculate_volatility_adjusted_size(
        base_size: float,
        current_volatility: float,
        historical_volatility: float,
        vol_target: float | None = 0.15,
    ) -> float:
        """
        Adjust position size based on volatility conditions.

        Volatility targeting: the position is scaled by a single ratio
        ``target / current_volatility`` (bounded to [0.1, 3.0]). When
        ``vol_target`` is None the asset's own ``historical_volatility`` is used
        as the target, i.e. the scaling is ``historical / current``.

        Args:
            base_size: Base position size
            current_volatility: Current asset volatility (annualized)
            historical_volatility: Historical average volatility (annualized);
                used as the target when ``vol_target`` is None
            vol_target: Target volatility level (default 15%), or None

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
        target = historical_volatility if vol_target is None else vol_target
        if target <= 0:
            raise ValueError("Volatility target must be positive")

        # Single volatility-targeting ratio, bounded to prevent extreme adjustments
        vol_adjustment = max(0.1, min(3.0, target / current_volatility))

        return _bound_position(base_size * vol_adjustment)

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
        vol_target: float | None = 0.15,
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
            historical_vol: Historical volatility (optional). Only used as the
                volatility target when ``vol_target`` is None.
            vol_target: Annualized volatility target for the volatility step
                (default 15%); None targets ``historical_vol`` instead

        Returns:
            Dictionary with various position size calculations. ``final_size``
            is 0.0 when any step finds no edge (e.g. UNKNOWN regime or
            non-positive Kelly).

        Raises:
            ValueError: If any input is invalid (no fallback size is returned)
        """
        results = {
            "base_size": base_size,
            "regime_adjusted": 0.0,
            "correlation_adjusted": 0.0,
            "kelly_optimal": 0.0,
            "volatility_adjusted": 0.0,
            "final_size": 0.0,
        }

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
        final_size = corr_adjusted
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly_size = SimonsRiskCalculator.calculate_kelly_optimal_size(
                win_rate, avg_win, avg_loss, confidence
            )
            results["kelly_optimal"] = kelly_size
            # Kelly caps the size; zero Kelly (no edge) means no position.
            # Re-apply the bounds so a tiny positive Kelly is floored consistently.
            final_size = _bound_position(min(corr_adjusted, kelly_size))

        # Volatility adjustment (if volatility data available)
        if current_vol is not None and historical_vol is not None:
            final_size = SimonsRiskCalculator.calculate_volatility_adjusted_size(
                final_size, current_vol, historical_vol, vol_target
            )
        results["volatility_adjusted"] = final_size
        results["final_size"] = final_size

        return results
