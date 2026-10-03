"""
Detector protocol and model-agnostic helpers shared by regime detectors.

:class:`RegimeDetector` is the interface :class:`~mra_lib.analyzer.MarketRegimeAnalyzer`
depends on, so any model implementing it can be injected (see the analyzer's
``detector_factory`` argument). The default implementation is
:class:`~mra_lib.indicators.true_hmm_detector.TrueHMMDetector`.
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd

from mra_lib.config.enums import MarketRegime

#: Fewer observations than this carry no information about persistence.
_MIN_PERSISTENCE_OBS = 2


@runtime_checkable
class RegimeDetector(Protocol):
    """Structural interface of a fitted-then-queried market regime detector."""

    n_states: int

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit the model on OHLCV data and return ``self``."""
        ...

    def predict_regime(self, df: pd.DataFrame) -> tuple[MarketRegime, int, float]:
        """Return ``(regime, state, confidence)`` for the last bar of ``df``."""
        ...

    def predict_with_states(self, df: pd.DataFrame) -> tuple[MarketRegime, np.ndarray, float]:
        """
        Return ``(regime, states, confidence)``.

        ``states`` holds one causal state estimate per feature row; its last
        element is the current state.
        """
        ...

    def get_transition_probability(self, from_state: int, to_state: int) -> float:
        """Probability of moving from ``from_state`` to ``to_state`` in one bar."""
        ...

    def calculate_regime_persistence(self, states: np.ndarray, lookback: int = 20) -> float:
        """Share of the last ``lookback`` states equal to the current one."""
        ...


def regime_persistence(states: np.ndarray, lookback: int = 20) -> float:
    """
    Fraction of the last ``lookback`` states equal to the current (last) state.

    Fewer than two observations carry no information about persistence and
    return ``0.0``.

    Args:
        states: State sequence (oldest first)
        lookback: Number of recent periods to examine

    Returns:
        Persistence score in ``[0, 1]`` (higher = more stable)
    """
    arr = np.asarray(states)
    lookback = min(lookback, len(arr))
    if lookback < _MIN_PERSISTENCE_OBS:
        return 0.0
    recent = arr[-lookback:]
    return float(np.mean(recent == recent[-1]))


def transition_probability(
    transition_matrix: np.ndarray | None, from_state: int, to_state: int
) -> float:
    """
    Look up ``transition_matrix[from_state, to_state]`` with validation.

    Raises:
        ValueError: If the matrix is missing (model not fitted) or an index is
            out of range
    """
    if transition_matrix is None:
        raise ValueError("Model must be fitted first")
    n_states = int(transition_matrix.shape[0])
    if not 0 <= from_state < n_states:
        raise ValueError(f"Invalid from_state: {from_state}")
    if not 0 <= to_state < n_states:
        raise ValueError(f"Invalid to_state: {to_state}")
    return float(transition_matrix[from_state, to_state])
