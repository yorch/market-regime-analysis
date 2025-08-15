"""
Data Provider Interface - Legacy Compatibility Module

This module provides backward compatibility by re-exporting the new
provider structure from the providers package.

For new code, import directly from: market_regime_analysis.providers
"""

# Import everything from the new providers package for backward compatibility
from .providers import (
    AlphaVantageProvider,
    MarketDataProvider,
    MockDataProvider,
    ProviderConfig,
    YFinanceProvider,
    create_provider,
    list_available_providers,
    register_provider,
)

# Export everything for backward compatibility
__all__ = [
    "AlphaVantageProvider",
    "MarketDataProvider",
    "MockDataProvider",
    "ProviderConfig",
    "YFinanceProvider",
    "create_provider",
    "list_available_providers",
    "register_provider",
]
