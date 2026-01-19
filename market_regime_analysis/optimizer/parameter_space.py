"""
Parameter space definition for optimization.

Defines the search space for all optimizable parameters including
regime multipliers, risk management settings, and HMM configuration.
"""

from dataclasses import dataclass, field
from typing import Any

from ..enums import MarketRegime


@dataclass
class ParameterRange:
    """Define a parameter's search range."""

    name: str
    min_value: float
    max_value: float
    step: float | None = None  # For grid search
    default: float | None = None

    def get_grid_values(self) -> list[float]:
        """Get discrete values for grid search."""
        if self.step is None:
            # Default: 10 values across range
            return [
                self.min_value + i * (self.max_value - self.min_value) / 9 for i in range(10)
            ]
        else:
            values = []
            current = self.min_value
            while current <= self.max_value:
                values.append(current)
                current += self.step
            return values


@dataclass
class ParameterSpace:
    """
    Complete parameter search space for strategy optimization.

    This class defines all optimizable parameters and their ranges,
    including regime multipliers, risk management settings, and
    HMM configuration parameters.
    """

    # Regime multipliers (default values from current implementation)
    regime_multipliers: dict[MarketRegime, ParameterRange] = field(default_factory=dict)

    # Risk management parameters
    stop_loss_pct: ParameterRange | None = None
    take_profit_pct: ParameterRange | None = None
    max_position_size: ParameterRange | None = None

    # HMM parameters
    volatility_threshold_high_pct: ParameterRange | None = None  # Percentile (e.g., 75)
    volatility_threshold_low_pct: ParameterRange | None = None  # Percentile (e.g., 25)

    def __post_init__(self):
        """Initialize default parameter ranges if not provided."""
        if not self.regime_multipliers:
            # Default regime multiplier ranges
            # Current values: BULL=1.3, BEAR=0.7, MEAN_REV=1.2, HIGH_VOL=0.4, LOW_VOL=1.1, BREAKOUT=0.9
            self.regime_multipliers = {
                MarketRegime.BULL_TRENDING: ParameterRange(
                    "bull_multiplier", min_value=0.8, max_value=1.8, step=0.1, default=1.3
                ),
                MarketRegime.BEAR_TRENDING: ParameterRange(
                    "bear_multiplier", min_value=0.3, max_value=1.0, step=0.1, default=0.7
                ),
                MarketRegime.MEAN_REVERTING: ParameterRange(
                    "mean_rev_multiplier", min_value=0.8, max_value=1.5, step=0.1, default=1.2
                ),
                MarketRegime.HIGH_VOLATILITY: ParameterRange(
                    "high_vol_multiplier", min_value=0.1, max_value=0.8, step=0.1, default=0.4
                ),
                MarketRegime.LOW_VOLATILITY: ParameterRange(
                    "low_vol_multiplier", min_value=0.8, max_value=1.5, step=0.1, default=1.1
                ),
                MarketRegime.BREAKOUT: ParameterRange(
                    "breakout_multiplier", min_value=0.5, max_value=1.5, step=0.1, default=0.9
                ),
                MarketRegime.UNKNOWN: ParameterRange(
                    "unknown_multiplier", min_value=0.1, max_value=0.5, step=0.1, default=0.2
                ),
            }

        if self.stop_loss_pct is None:
            self.stop_loss_pct = ParameterRange(
                "stop_loss_pct", min_value=0.05, max_value=0.15, step=0.01, default=0.10
            )

        if self.take_profit_pct is None:
            self.take_profit_pct = ParameterRange(
                "take_profit_pct", min_value=0.10, max_value=0.30, step=0.05, default=0.15
            )

        if self.max_position_size is None:
            self.max_position_size = ParameterRange(
                "max_position_size", min_value=0.10, max_value=0.30, step=0.05, default=0.20
            )

        if self.volatility_threshold_high_pct is None:
            self.volatility_threshold_high_pct = ParameterRange(
                "vol_threshold_high",
                min_value=60.0,
                max_value=90.0,
                step=5.0,
                default=75.0,
            )

        if self.volatility_threshold_low_pct is None:
            self.volatility_threshold_low_pct = ParameterRange(
                "vol_threshold_low",
                min_value=10.0,
                max_value=40.0,
                step=5.0,
                default=25.0,
            )

    def get_default_parameters(self) -> dict[str, Any]:
        """Get default parameter values."""
        params = {}

        # Regime multipliers
        for regime, param_range in self.regime_multipliers.items():
            params[f"regime_mult_{regime.value}"] = param_range.default

        # Risk management
        params["stop_loss_pct"] = self.stop_loss_pct.default
        params["take_profit_pct"] = self.take_profit_pct.default
        params["max_position_size"] = self.max_position_size.default

        # HMM
        params["vol_threshold_high_pct"] = self.volatility_threshold_high_pct.default
        params["vol_threshold_low_pct"] = self.volatility_threshold_low_pct.default

        return params

    def count_total_combinations(self) -> int:
        """
        Count total number of parameter combinations for grid search.

        Warning: Can be very large! Consider using subset optimization.
        """
        count = 1

        # Regime multipliers
        for param_range in self.regime_multipliers.values():
            count *= len(param_range.get_grid_values())

        # Risk management
        count *= len(self.stop_loss_pct.get_grid_values())
        count *= len(self.take_profit_pct.get_grid_values())
        count *= len(self.max_position_size.get_grid_values())

        # HMM
        count *= len(self.volatility_threshold_high_pct.get_grid_values())
        count *= len(self.volatility_threshold_low_pct.get_grid_values())

        return count

    def get_subset_space(self, optimize_regime_only: bool = False) -> "ParameterSpace":
        """
        Get a reduced parameter space for faster optimization.

        Args:
            optimize_regime_only: If True, only optimize regime multipliers

        Returns:
            Reduced ParameterSpace
        """
        if optimize_regime_only:
            # Keep regime multipliers, use defaults for others
            return ParameterSpace(
                regime_multipliers=self.regime_multipliers,
                stop_loss_pct=ParameterRange(
                    "stop_loss_pct",
                    min_value=0.10,
                    max_value=0.10,
                    step=1.0,
                    default=0.10,
                ),
                take_profit_pct=ParameterRange(
                    "take_profit_pct",
                    min_value=0.15,
                    max_value=0.15,
                    step=1.0,
                    default=0.15,
                ),
                max_position_size=ParameterRange(
                    "max_position_size",
                    min_value=0.20,
                    max_value=0.20,
                    step=1.0,
                    default=0.20,
                ),
                volatility_threshold_high_pct=ParameterRange(
                    "vol_threshold_high",
                    min_value=75.0,
                    max_value=75.0,
                    step=1.0,
                    default=75.0,
                ),
                volatility_threshold_low_pct=ParameterRange(
                    "vol_threshold_low",
                    min_value=25.0,
                    max_value=25.0,
                    step=1.0,
                    default=25.0,
                ),
            )
        else:
            # Return full space (default)
            return self
