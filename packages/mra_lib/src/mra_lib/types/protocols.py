"""
Protocol definitions for dependency injection.

The core library defines these protocols so that external packages
(CLI, web, bots) can plug in their own implementations without the
library depending on any specific framework.
"""

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.enums import MarketRegime


@runtime_checkable
class DataStoreProtocol(Protocol):
    """Protocol for persisting analysis results."""

    def save_analysis(self, symbol: str, timeframe: str, analysis: RegimeAnalysis) -> None:
        """Persist a single regime analysis result."""
        ...

    def load_analysis(self, symbol: str, timeframe: str) -> RegimeAnalysis | None:
        """Load the most recent analysis for a symbol/timeframe pair."""
        ...

    def list_symbols(self) -> list[str]:
        """Return all symbols that have stored analyses."""
        ...


@runtime_checkable
class DashboardProtocol(Protocol):
    """Protocol for displaying analysis results to users."""

    def display_regime(self, symbol: str, regime: MarketRegime, confidence: float) -> None:
        """Display the current regime for a symbol."""
        ...

    def display_analysis(self, symbol: str, analysis: RegimeAnalysis) -> None:
        """Display a full analysis result."""
        ...

    def display_error(self, message: str) -> None:
        """Display an error message."""
        ...


@runtime_checkable
class MarketDataProviderProtocol(Protocol):
    """Protocol for fetching market data — decouples lib from concrete providers."""

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch OHLCV data for the given symbol, period, and interval."""
        ...

    @classmethod
    def get_available_providers(cls) -> dict[str, dict[str, Any]]:
        """Return metadata about all registered providers."""
        ...
