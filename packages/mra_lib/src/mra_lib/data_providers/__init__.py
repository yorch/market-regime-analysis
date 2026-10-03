"""
Market Data Providers Package

Plug-and-play architecture for market data providers with automatic provider discovery.
"""

from .alpaca_provider import AlpacaProvider
from .alphavantage_provider import AlphaVantageProvider
from .base import MarketDataProvider, ProviderConfig
from .credentials import required_env_vars, requires_credentials, resolve_api_key
from .mock_provider import MockDataProvider
from .polygon_provider import PolygonProvider
from .tiingo_provider import TiingoProvider
from .yfinance_provider import YFinanceProvider

# Auto-register core providers
MarketDataProvider.register(YFinanceProvider)
MarketDataProvider.register(AlphaVantageProvider)
MarketDataProvider.register(PolygonProvider)
MarketDataProvider.register(AlpacaProvider)
MarketDataProvider.register(TiingoProvider)

__all__ = [
    "AlpacaProvider",
    "AlphaVantageProvider",
    "MarketDataProvider",
    "MockDataProvider",
    "PolygonProvider",
    "ProviderConfig",
    "TiingoProvider",
    "YFinanceProvider",
    "required_env_vars",
    "requires_credentials",
    "resolve_api_key",
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
