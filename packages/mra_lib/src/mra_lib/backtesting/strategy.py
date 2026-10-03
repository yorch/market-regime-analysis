"""
Configurable regime-based trading strategy.

Separates strategy logic from the backtest engine so parameters
can be optimized independently.
"""

from collections.abc import Iterable

import pandas as pd

from mra_lib.config.enums import MarketRegime, TradingStrategy
from mra_lib.config.regime_tables import REGIME_MULTIPLIERS

#: Keys accepted by :meth:`RegimeStrategy.from_param_vector`.
PARAM_KEYS = frozenset(
    {
        "bull_mult",
        "bear_mult",
        "mr_mult",
        "hv_mult",
        "lv_mult",
        "bo_mult",
        "base_fraction",
        "max_position",
        "stop_loss",
        "take_profit",
        "min_confidence",
        "confidence_scaling",
        "bear_short",
    }
)


class RegimeStrategy:
    """
    Parameterized regime-based trading strategy.

    All tunable parameters are exposed for optimization.
    """

    def __init__(
        self,
        regime_multipliers: dict[MarketRegime, float] | None = None,
        regime_directions: dict[MarketRegime, str | None] | None = None,
        base_position_fraction: float = 0.10,
        max_position_size: float = 0.20,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float | None = None,
        min_confidence: float = 0.0,
        confidence_scaling: bool = True,
    ) -> None:
        """
        Initialize strategy with tunable parameters.

        Args:
            regime_multipliers: Position size multiplier per regime
            regime_directions: Trade direction per regime (LONG/SHORT/None)
            base_position_fraction: Base fraction of capital per trade
            max_position_size: Max fraction of capital per position
            stop_loss_pct: Stop loss as fraction of entry price
            take_profit_pct: Take profit as fraction (None = disabled)
            min_confidence: Minimum regime confidence to enter trades
            confidence_scaling: Scale position size by confidence
        """
        self.regime_multipliers = regime_multipliers or dict(REGIME_MULTIPLIERS)

        self.regime_directions = regime_directions or {
            MarketRegime.BULL_TRENDING: "LONG",
            MarketRegime.BEAR_TRENDING: "SHORT",
            MarketRegime.MEAN_REVERTING: "LONG",
            MarketRegime.LOW_VOLATILITY: "LONG",
            MarketRegime.HIGH_VOLATILITY: None,  # No trade
            MarketRegime.BREAKOUT: "LONG",
            MarketRegime.UNKNOWN: None,  # No trade
        }

        self.base_position_fraction = base_position_fraction
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_confidence = min_confidence
        self.confidence_scaling = confidence_scaling

    def get_signal(self, regime: MarketRegime, confidence: float) -> tuple[str | None, float]:
        """
        Get trading signal for current regime.

        Returns:
            Tuple of (direction, position_multiplier).
            direction is None if no trade should be taken.
        """
        if confidence < self.min_confidence:
            return None, 0.0

        direction = self.regime_directions.get(regime)
        multiplier = self.regime_multipliers.get(regime, 0.0)

        if direction is None or multiplier <= 0:
            return None, 0.0

        if self.confidence_scaling:
            multiplier *= confidence

        # Scale by base position fraction
        multiplier *= self.base_position_fraction

        return direction, multiplier

    def generate_signals(
        self,
        regimes: pd.Series,
        confidences: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate full signal series from regime predictions.

        Returns:
            Tuple of (strategies, directions, position_sizes) series.
        """
        strategies = []
        directions: list[str | None] = []
        position_sizes = []

        # NOTE: intentionally differs from config.regime_tables.REGIME_STRATEGIES
        # (which maps BEAR_TRENDING -> DEFENSIVE and HIGH_VOLATILITY ->
        # VOLATILITY_TRADING). The engine closes positions on DEFENSIVE/AVOID, so
        # adopting the shared map would make every bear-regime short exit
        # immediately and change backtest behavior. Unifying the maps is an open
        # decision, kept separate from accounting fixes.
        regime_to_strategy = {
            MarketRegime.BULL_TRENDING: TradingStrategy.TREND_FOLLOWING,
            MarketRegime.BEAR_TRENDING: TradingStrategy.TREND_FOLLOWING,
            MarketRegime.MEAN_REVERTING: TradingStrategy.MEAN_REVERSION,
            MarketRegime.LOW_VOLATILITY: TradingStrategy.MOMENTUM,
            MarketRegime.HIGH_VOLATILITY: TradingStrategy.DEFENSIVE,
            MarketRegime.BREAKOUT: TradingStrategy.MOMENTUM,
            MarketRegime.UNKNOWN: TradingStrategy.AVOID,
        }

        for i in range(len(regimes)):
            regime = regimes.iloc[i]
            conf = confidences.iloc[i] if i < len(confidences) else 0.5

            direction, mult = self.get_signal(regime, conf)

            if direction is None:
                strategies.append(TradingStrategy.AVOID)
                directions.append(None)
                position_sizes.append(0.0)
            else:
                strategies.append(regime_to_strategy.get(regime, TradingStrategy.AVOID))
                directions.append(direction)
                position_sizes.append(mult)

        return (
            pd.Series(strategies, index=regimes.index),
            pd.Series(directions, index=regimes.index),
            pd.Series(position_sizes, index=regimes.index),
        )

    def to_dict(self) -> dict:
        """Serialize strategy parameters for logging."""
        return {
            "regime_multipliers": {k.value: v for k, v in self.regime_multipliers.items()},
            "regime_directions": {k.value: v for k, v in self.regime_directions.items()},
            "base_position_fraction": self.base_position_fraction,
            "max_position_size": self.max_position_size,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "min_confidence": self.min_confidence,
            "confidence_scaling": self.confidence_scaling,
        }

    @staticmethod
    def validate_param_keys(keys: Iterable[str]) -> None:
        """
        Raise ``ValueError`` if any key is not a recognised strategy parameter.

        Catches typos (e.g. ``stoploss``) that would otherwise be silently
        ignored and leave the default in place.
        """
        unknown = sorted(set(keys) - PARAM_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown strategy parameter(s): {', '.join(unknown)}. "
                f"Valid keys: {', '.join(sorted(PARAM_KEYS))}"
            )

    @staticmethod
    def canonical_params(params: dict) -> dict:
        """
        Return an equivalent parameter dict with irrelevant settings normalised.

        A bear position is only taken when ``bear_short`` is truthy *and*
        ``bear_mult > 0``; otherwise both settings are equivalent to
        ``bear_mult=0, bear_short=0``. Used by the optimizer to skip combinations
        that would produce identical backtests.
        """
        canon = dict(params)
        bear_mult = canon.get("bear_mult", REGIME_MULTIPLIERS[MarketRegime.BEAR_TRENDING])
        bear_short = canon.get("bear_short", 1)
        if not bear_short or bear_mult <= 0:
            canon["bear_mult"] = 0.0
            canon["bear_short"] = 0
        return canon

    @staticmethod
    def from_param_vector(params: dict) -> "RegimeStrategy":
        """
        Create strategy from a flat parameter dictionary.

        Expected keys:
            bull_mult, bear_mult, mr_mult, hv_mult, lv_mult, bo_mult,
            base_fraction, max_position, stop_loss, take_profit,
            min_confidence, confidence_scaling, bear_short (1=SHORT, 0=None)

        Raises:
            ValueError: If ``params`` contains an unknown key
        """
        RegimeStrategy.validate_param_keys(params)
        defaults = REGIME_MULTIPLIERS
        regime_multipliers = {
            MarketRegime.BULL_TRENDING: params.get(
                "bull_mult", defaults[MarketRegime.BULL_TRENDING]
            ),
            MarketRegime.BEAR_TRENDING: params.get(
                "bear_mult", defaults[MarketRegime.BEAR_TRENDING]
            ),
            MarketRegime.MEAN_REVERTING: params.get(
                "mr_mult", defaults[MarketRegime.MEAN_REVERTING]
            ),
            # HIGH_VOLATILITY has no trade direction, so its multiplier is inert;
            # 0.0 keeps the historical default for this flat parameter vector.
            MarketRegime.HIGH_VOLATILITY: params.get("hv_mult", 0.0),
            MarketRegime.LOW_VOLATILITY: params.get(
                "lv_mult", defaults[MarketRegime.LOW_VOLATILITY]
            ),
            MarketRegime.BREAKOUT: params.get("bo_mult", defaults[MarketRegime.BREAKOUT]),
            MarketRegime.UNKNOWN: 0.0,
        }

        bear_dir = "SHORT" if params.get("bear_short", 1) else None
        regime_directions = {
            MarketRegime.BULL_TRENDING: "LONG",
            MarketRegime.BEAR_TRENDING: bear_dir,
            MarketRegime.MEAN_REVERTING: "LONG",
            MarketRegime.LOW_VOLATILITY: "LONG",
            MarketRegime.HIGH_VOLATILITY: None,
            MarketRegime.BREAKOUT: "LONG",
            MarketRegime.UNKNOWN: None,
        }

        return RegimeStrategy(
            regime_multipliers=regime_multipliers,
            regime_directions=regime_directions,
            base_position_fraction=params.get("base_fraction", 0.10),
            max_position_size=params.get("max_position", 0.20),
            stop_loss_pct=params.get("stop_loss", 0.05),
            take_profit_pct=params.get("take_profit"),
            min_confidence=params.get("min_confidence", 0.0),
            confidence_scaling=params.get("confidence_scaling", True),
        )
