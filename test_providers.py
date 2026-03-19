"""Tests for data provider base, mock provider, and registry."""

import pandas as pd
import pytest

from market_regime_analysis.providers.base import MarketDataProvider, ProviderConfig
from market_regime_analysis.providers.mock_provider import MockDataProvider

# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_defaults(self):
        config = ProviderConfig()
        assert config.api_key is None
        assert config.timeout == 30
        assert config.retries == 3
        assert config.rate_limit == 0.0

    def test_custom_values(self):
        config = ProviderConfig(api_key="test_key", timeout=60)
        assert config.api_key == "test_key"
        assert config.timeout == 60

    def test_extra_kwargs_stored(self):
        config = ProviderConfig(custom_field="hello")
        assert config.custom_field == "hello"


# ---------------------------------------------------------------------------
# MarketDataProvider registry
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    """Tests for provider registration and factory."""

    def test_core_providers_registered(self):
        providers = MarketDataProvider.get_available_providers()
        assert "yfinance" in providers
        assert "alphavantage" in providers
        assert "polygon" in providers

    def test_get_available_providers_returns_metadata(self):
        providers = MarketDataProvider.get_available_providers()
        for _name, info in providers.items():
            assert "description" in info
            assert "requires_api_key" in info
            assert "supported_intervals" in info
            assert "supported_periods" in info
            assert "rate_limit_per_minute" in info

    def test_create_provider_yfinance(self):
        provider = MarketDataProvider.create_provider("yfinance")
        assert provider is not None
        assert provider.provider_name == "yfinance"

    def test_create_provider_unknown_raises(self):
        with pytest.raises(ValueError, match="not found"):
            MarketDataProvider.create_provider("nonexistent_provider")

    def test_register_provider_without_name_raises(self):
        class BadProvider(MarketDataProvider):
            provider_name = ""

            def fetch(self, symbol, period, interval):
                return pd.DataFrame()

        with pytest.raises(ValueError, match="must define provider_name"):
            MarketDataProvider.register(BadProvider)

    def test_register_and_create_custom_provider(self):
        from typing import ClassVar

        class TestProvider(MarketDataProvider):
            provider_name = "test_custom"
            supported_intervals: ClassVar[set[str]] = {"1d"}
            supported_periods: ClassVar[set[str]] = {"1y"}
            requires_api_key = False
            description = "Test provider"

            def fetch(self, symbol, period, interval):
                return pd.DataFrame(
                    {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]},
                    index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]),
                )

        MarketDataProvider.register(TestProvider)
        try:
            provider = MarketDataProvider.create_provider("test_custom")
            assert provider.provider_name == "test_custom"
        finally:
            # Clean up registry
            MarketDataProvider._providers.pop("test_custom", None)

    def test_api_key_required_validation(self):
        class KeyProvider(MarketDataProvider):
            provider_name = "key_test"
            requires_api_key = True

            def fetch(self, symbol, period, interval):
                return pd.DataFrame()

        # Should raise because no api_key provided
        with pytest.raises(ValueError, match="requires an API key"):
            KeyProvider()

    def test_api_key_provided_no_error(self):
        class KeyProvider(MarketDataProvider):
            provider_name = "key_test2"
            requires_api_key = True

            def fetch(self, symbol, period, interval):
                return pd.DataFrame()

        config = ProviderConfig(api_key="my_key")
        provider = KeyProvider(config)
        assert provider.config.api_key == "my_key"


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestValidateParameters:
    """Tests for validate_parameters."""

    def test_valid_params_no_error(self):
        provider = MockDataProvider()
        provider.validate_parameters("SPY", "1y", "1d")

    def test_invalid_interval_raises(self):
        provider = MockDataProvider()
        with pytest.raises(ValueError, match="Interval.*not supported"):
            provider.validate_parameters("SPY", "1y", "1w")

    def test_invalid_period_raises(self):
        provider = MockDataProvider()
        with pytest.raises(ValueError, match="Period.*not supported"):
            provider.validate_parameters("SPY", "10y", "1d")


# ---------------------------------------------------------------------------
# standardize_dataframe
# ---------------------------------------------------------------------------


class TestStandardizeDataframe:
    """Tests for standardize_dataframe."""

    def test_valid_df_passes(self):
        provider = MockDataProvider()
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]),
        )
        result = provider.standardize_dataframe(df)
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_missing_column_raises(self):
        provider = MockDataProvider()
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Close": [1.5], "Volume": [100]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]),
        )
        with pytest.raises(ValueError, match="Missing required column: Low"):
            provider.standardize_dataframe(df)

    def test_non_datetime_index_converted(self):
        provider = MockDataProvider()
        df = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]},
            index=["2024-01-01"],
        )
        result = provider.standardize_dataframe(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_sorts_by_date(self):
        provider = MockDataProvider()
        df = pd.DataFrame(
            {
                "Open": [1, 2],
                "High": [2, 3],
                "Low": [0.5, 1],
                "Close": [1.5, 2.5],
                "Volume": [100, 200],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")]),
        )
        result = provider.standardize_dataframe(df)
        assert result.index[0] < result.index[1]

    def test_drops_all_nan_rows(self):
        provider = MockDataProvider()
        df = pd.DataFrame(
            {
                "Open": [1, float("nan")],
                "High": [2, float("nan")],
                "Low": [0.5, float("nan")],
                "Close": [1.5, float("nan")],
                "Volume": [100, float("nan")],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]),
        )
        result = provider.standardize_dataframe(df)
        assert len(result) == 1

    def test_extra_columns_stripped(self):
        provider = MockDataProvider()
        df = pd.DataFrame(
            {
                "Open": [1],
                "High": [2],
                "Low": [0.5],
                "Close": [1.5],
                "Volume": [100],
                "Extra": [999],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]),
        )
        result = provider.standardize_dataframe(df)
        assert "Extra" not in result.columns


# ---------------------------------------------------------------------------
# MockDataProvider
# ---------------------------------------------------------------------------


class TestMockDataProvider:
    """Tests for the mock data provider."""

    def test_provider_name(self):
        provider = MockDataProvider()
        assert provider.provider_name == "mock"

    def test_no_api_key_required(self):
        provider = MockDataProvider()
        assert not provider.requires_api_key

    def test_fetch_returns_dataframe(self):
        provider = MockDataProvider()
        df = provider.fetch("SPY", "1y", "1d")
        assert isinstance(df, pd.DataFrame)

    def test_fetch_has_ohlcv_columns(self):
        provider = MockDataProvider()
        df = provider.fetch("SPY", "1y", "1d")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns

    def test_fetch_daily_has_252_bars(self):
        provider = MockDataProvider()
        df = provider.fetch("SPY", "1y", "1d")
        assert len(df) == 252

    def test_fetch_2y_has_504_bars(self):
        provider = MockDataProvider()
        df = provider.fetch("SPY", "2y", "1d")
        assert len(df) == 504

    def test_fetch_datetime_index(self):
        provider = MockDataProvider()
        df = provider.fetch("SPY", "1y", "1d")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_high_greater_than_low(self):
        provider = MockDataProvider()
        df = provider.fetch("SPY", "1y", "1d")
        assert (df["High"] >= df["Low"]).all()

    def test_volume_positive(self):
        provider = MockDataProvider()
        df = provider.fetch("SPY", "1y", "1d")
        assert (df["Volume"] > 0).all()

    def test_close_positive(self):
        provider = MockDataProvider()
        df = provider.fetch("SPY", "1y", "1d")
        assert (df["Close"] > 0).all()

    def test_invalid_interval_raises(self):
        provider = MockDataProvider()
        with pytest.raises(ValueError):
            provider.fetch("SPY", "1y", "3d")

    def test_invalid_period_raises(self):
        provider = MockDataProvider()
        with pytest.raises(ValueError):
            provider.fetch("SPY", "5y", "1d")

    def test_reproducible_data(self):
        """Mock data values should be reproducible (uses seed=42)."""
        p1 = MockDataProvider()
        p2 = MockDataProvider()
        df1 = p1.fetch("SPY", "1y", "1d")
        df2 = p2.fetch("SPY", "1y", "1d")
        # Index uses datetime.now() so timestamps differ; compare values only
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True), df2.reset_index(drop=True)
        )
