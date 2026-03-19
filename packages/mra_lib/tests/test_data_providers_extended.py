"""Extended tests for data_providers — init helpers, alphavantage, polygon."""

from typing import ClassVar

import pandas as pd
import pytest

from mra_lib.data_providers import (
    create_provider,
    list_available_providers,
    register_provider,
)
from mra_lib.data_providers.base import MarketDataProvider, ProviderConfig


class TestModuleLevelHelpers:
    def test_list_available_providers(self):
        providers = list_available_providers()
        assert isinstance(providers, dict)
        assert "yfinance" in providers
        assert "alphavantage" in providers
        assert "polygon" in providers

    def test_create_provider_yfinance(self):
        provider = create_provider("yfinance")
        assert isinstance(provider, MarketDataProvider)

    def test_create_provider_unknown_raises(self):
        with pytest.raises(ValueError):
            create_provider("nonexistent_provider")

    def test_register_custom_provider(self):
        class CustomProvider(MarketDataProvider):
            provider_name = "custom_test_provider_42"
            description = "Test"
            requires_api_key = False
            rate_limit_per_minute = 100
            supported_intervals: ClassVar[set[str]] = {"1d"}
            supported_periods: ClassVar[set[str]] = {"1y"}

            def fetch(self, symbol, period, interval):
                return pd.DataFrame()

        register_provider(CustomProvider)
        providers = list_available_providers()
        assert "custom_test_provider_42" in providers
        # Clean up
        if "custom_test_provider_42" in MarketDataProvider._providers:
            del MarketDataProvider._providers["custom_test_provider_42"]


class TestAlphaVantageProvider:
    def test_provider_name(self):
        from mra_lib.data_providers.alphavantage_provider import AlphaVantageProvider

        assert AlphaVantageProvider.provider_name == "alphavantage"
        assert AlphaVantageProvider.requires_api_key is True

    def test_init_with_api_key(self):
        from mra_lib.data_providers.alphavantage_provider import AlphaVantageProvider

        config = ProviderConfig(api_key="test_key")
        try:
            provider = AlphaVantageProvider(config=config)
            assert provider.config.api_key == "test_key"
        except ImportError:
            pytest.skip("alpha_vantage not installed")

    def test_supported_intervals(self):
        from mra_lib.data_providers.alphavantage_provider import AlphaVantageProvider

        assert "1d" in AlphaVantageProvider.supported_intervals
        assert "1h" in AlphaVantageProvider.supported_intervals

    def test_supported_periods(self):
        from mra_lib.data_providers.alphavantage_provider import AlphaVantageProvider

        assert "1y" in AlphaVantageProvider.supported_periods
        assert "2y" in AlphaVantageProvider.supported_periods


class TestPolygonProvider:
    def test_provider_name(self):
        from mra_lib.data_providers.polygon_provider import PolygonProvider

        assert PolygonProvider.provider_name == "polygon"
        assert PolygonProvider.requires_api_key is True

    def test_init_with_api_key(self):
        from mra_lib.data_providers.polygon_provider import PolygonProvider

        config = ProviderConfig(api_key="test_key")
        provider = PolygonProvider(config=config)
        assert provider.config.api_key == "test_key"

    def test_supported_intervals(self):
        from mra_lib.data_providers.polygon_provider import PolygonProvider

        assert "1d" in PolygonProvider.supported_intervals

    def test_supported_periods(self):
        from mra_lib.data_providers.polygon_provider import PolygonProvider

        assert "1y" in PolygonProvider.supported_periods


class TestYFinanceProvider:
    def test_provider_name(self):
        from mra_lib.data_providers.yfinance_provider import YFinanceProvider

        assert YFinanceProvider.provider_name == "yfinance"
        assert YFinanceProvider.requires_api_key is False

    def test_init_no_key_needed(self):
        from mra_lib.data_providers.yfinance_provider import YFinanceProvider

        YFinanceProvider()  # Should not raise


class TestProviderConfig:
    def test_defaults(self):
        config = ProviderConfig()
        assert config.api_key is None
        assert config.timeout == 30
        assert config.retries == 3

    def test_custom_values(self):
        config = ProviderConfig(api_key="key", timeout=60, retries=5)
        assert config.api_key == "key"
        assert config.timeout == 60
        assert config.retries == 5
