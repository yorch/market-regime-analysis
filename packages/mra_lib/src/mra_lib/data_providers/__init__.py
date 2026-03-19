"""
Market Data Providers Package

Plug-and-play architecture for market data providers with automatic provider discovery.
"""

from .alphavantage_provider import AlphaVantageProvider
from .base import MarketDataProvider, ProviderConfig
from .mock_provider import MockDataProvider
from .polygon_provider import PolygonProvider
from .yfinance_provider import YFinanceProvider

# Auto-register core providers
MarketDataProvider.register(YFinanceProvider)
MarketDataProvider.register(AlphaVantageProvider)
MarketDataProvider.register(PolygonProvider)

__all__ = [
    "AlphaVantageProvider",
    "MarketDataProvider",
    "MockDataProvider",
    "PolygonProvider",
    "ProviderConfig",
    "YFinanceProvider",
]


def list_available_providers() -> dict[str, dict]:
    """Convenience function to list all available providers."""
    return MarketDataProvider.get_available_providers()


def create_provider(provider_name: str, **config_kwargs) -> MarketDataProvider:
    """Convenience function to create a provider instance."""
    return MarketDataProvider.create_provider(provider_name, **config_kwargs)


def register_provider(provider_class: type[MarketDataProvider]) -> None:
    """Register a new provider class."""
    MarketDataProvider.register(provider_class)
