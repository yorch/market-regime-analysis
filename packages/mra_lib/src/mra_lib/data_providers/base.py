"""
Base Provider Interface

Core interfaces and utilities for market data providers in the plug-and-play architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd


class ProviderConfig:
    """Configuration container for data provider settings."""

    def __init__(self, **kwargs: Any) -> None:
        self.api_key: str | None = kwargs.get("api_key")
        self.timeout: int = kwargs.get("timeout", 30)
        self.retries: int = kwargs.get("retries", 3)
        self.rate_limit: float = kwargs.get("rate_limit", 0.0)

        # Store any additional provider-specific config
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class MarketDataProvider(ABC):
    """
    Abstract base class for market data providers.

    This class defines the interface that all market data providers must implement
    and provides common functionality for registration and validation.
    """

    # Registry for all available providers
    _providers: ClassVar[dict[str, type["MarketDataProvider"]]] = {}

    # Provider metadata
    provider_name: str = ""
    supported_intervals: ClassVar[set[str]] = set()
    supported_periods: ClassVar[set[str]] = set()
    requires_api_key: bool = False
    rate_limit_per_minute: int = 0
    description: str = ""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize provider with configuration."""
        self.config = config or ProviderConfig()
        self._validate_config()

    @abstractmethod
    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetch market data for a symbol, period, and interval.

        Args:
            symbol: Trading symbol (e.g., 'SPY', 'AAPL')
            period: Time period for data (e.g., '1y', '2y', '6mo')
            interval: Data interval (e.g., '1d', '1h', '15m')

        Returns:
            DataFrame with standardized OHLCV columns:
            - Open, High, Low, Close, Volume
            - Index should be datetime

        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If API is unavailable
            Exception: For provider-specific errors
        """
        pass

    @classmethod
    def register(cls, provider_class: type["MarketDataProvider"]) -> None:
        """Register a provider class in the global registry."""
        if not provider_class.provider_name:
            raise ValueError(f"Provider {provider_class.__name__} must define provider_name")

        cls._providers[provider_class.provider_name] = provider_class

    @classmethod
    def get_available_providers(cls) -> dict[str, dict[str, Any]]:
        """Get information about all available providers."""
        return {
            name: {
                "class": provider_class,
                "description": provider_class.description,
                "requires_api_key": provider_class.requires_api_key,
                "supported_intervals": list(provider_class.supported_intervals),
                "supported_periods": list(provider_class.supported_periods),
                "rate_limit_per_minute": provider_class.rate_limit_per_minute,
            }
            for name, provider_class in cls._providers.items()
        }

    @classmethod
    def create_provider(cls, provider_name: str, **config_kwargs: Any) -> "MarketDataProvider":
        """
        Factory method to create a provider instance.

        Args:
            provider_name: Name of the provider to create
            **config_kwargs: Configuration parameters for the provider

        Returns:
            Configured provider instance

        Raises:
            ValueError: If provider is not registered
        """
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Provider '{provider_name}' not found. Available: {available}")

        provider_class = cls._providers[provider_name]
        config = ProviderConfig(**config_kwargs)
        return provider_class(config)

    def _validate_config(self) -> None:
        """Validate provider configuration."""
        if self.requires_api_key and not self.config.api_key:
            raise ValueError(f"Provider '{self.provider_name}' requires an API key")

    def validate_parameters(self, symbol: str, period: str, interval: str) -> None:
        """
        Validate input parameters against provider capabilities.

        Args:
            symbol: Trading symbol
            period: Time period
            interval: Data interval

        Raises:
            ValueError: If parameters are not supported
        """
        if self.supported_intervals and interval not in self.supported_intervals:
            raise ValueError(
                f"Interval '{interval}' not supported by {self.provider_name}. "
                f"Supported: {sorted(self.supported_intervals)}"
            )

        if self.supported_periods and period not in self.supported_periods:
            raise ValueError(
                f"Period '{period}' not supported by {self.provider_name}. "
                f"Supported: {sorted(self.supported_periods)}"
            )

    def standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame format across providers.

        Args:
            df: Raw DataFrame from provider

        Returns:
            Standardized DataFrame with OHLCV columns
        """
        # Ensure we have the required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by date
        df = df.sort_index()

        # Remove any rows with all NaN values
        df = df.dropna(how="all")

        return df[required_columns]
