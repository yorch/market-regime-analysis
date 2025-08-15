"""
Market Data Providers Package

Plug-and-play architecture for market data providers with automatic provider discovery.
"""

# Import and auto-register all available providers
from .alphavantage_provider import AlphaVantageProvider
from .base import MarketDataProvider, ProviderConfig

# Import mock provider but don't auto-register (for development/testing only)
from .mock_provider import MockDataProvider
from .yfinance_provider import YFinanceProvider

# Auto-register core providers
MarketDataProvider.register(YFinanceProvider)
MarketDataProvider.register(AlphaVantageProvider)

# Uncomment the line below to enable mock provider for testing:
# MarketDataProvider.register(MockDataProvider)

# Export the main interfaces
__all__ = [
    "AlphaVantageProvider",
    "MarketDataProvider",
    "MockDataProvider",
    "ProviderConfig",
    "YFinanceProvider",
]


def list_available_providers() -> dict[str, dict]:
    """
    Convenience function to list all available providers.

    Returns:
        Dictionary with provider information
    """
    return MarketDataProvider.get_available_providers()


def create_provider(provider_name: str, **config_kwargs) -> MarketDataProvider:
    """
    Convenience function to create a provider instance.

    Args:
        provider_name: Name of the provider to create
        **config_kwargs: Configuration parameters

    Returns:
        Configured provider instance
    """
    return MarketDataProvider.create_provider(provider_name, **config_kwargs)


def register_provider(provider_class: type[MarketDataProvider]) -> None:
    """
    Register a new provider class.

    Args:
        provider_class: Provider class to register
    """
    MarketDataProvider.register(provider_class)
