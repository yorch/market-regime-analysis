"""
Polygon.io Data Provider

Professional market data provider with high-quality tick data and aggregates.
Provides real-time and historical data with excellent coverage and reliability.
"""

import re
from datetime import date, datetime, timedelta
from typing import ClassVar

import pandas as pd

from .base import MarketDataProvider, ProviderConfig


class PolygonProvider(MarketDataProvider):
    """Polygon.io data provider with API key authentication."""

    provider_name = "polygon"

    # Constants for chunking strategy
    _MINUTE_DATA_CHUNK_DAYS = 30  # Maximum days per chunk for minute data
    supported_intervals: ClassVar[set[str]] = {
        # Polygon.io native timespan values
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "quarter",
        "year",
        # Common alternative formats that map to Polygon.io
        "1m",  # maps to 1/minute
        "5m",  # maps to 5/minute
        "15m",  # maps to 15/minute
        "30m",  # maps to 30/minute
        "1h",  # maps to 1/hour
        "1hour",  # maps to 1/hour
        "1d",  # maps to 1/day
        "1day",  # maps to 1/day
        "daily",  # maps to 1/day
        "1w",  # maps to 1/week
        "1wk",  # maps to 1/week
        "1week",  # maps to 1/week
        "weekly",  # maps to 1/week
        "1mo",  # maps to 1/month
        "1month",  # maps to 1/month
        "monthly",  # maps to 1/month
    }
    supported_periods: ClassVar[set[str]] = {
        # Standard period formats
        "1d",  # 1 day
        "5d",  # 5 days
        "1mo",  # 1 month
        "2mo",  # 2 months
        "3mo",  # 3 months
        "6mo",  # 6 months
        "1y",  # 1 year
        "2y",  # 2 years
        "5y",  # 5 years
        "10y",  # 10 years
        "max",  # Maximum available data
        "ytd",  # Year to date
    }
    requires_api_key = True
    rate_limit_per_minute = 60  # Basic tier: 5 calls/minute, Pro: unlimited
    description = "High-quality market data from Polygon.io with tick-level precision"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize Polygon.io provider with API key."""
        super().__init__(config)

        try:
            from polygon import RESTClient  # noqa: PLC0415

            self.client: RESTClient = RESTClient(api_key=self.config.api_key)
        except ImportError as e:
            raise ImportError(
                "polygon-api-client library is required. Install with: pip install polygon-api-client"
            ) from e

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetch data from Polygon.io API.

        Args:
            symbol: Trading symbol (e.g., 'SPY', 'AAPL')
            period: Time period for data (e.g., '1y', '2y', '6mo')
            interval: Data interval (e.g., '1d', '1h', '15m')

        Returns:
            DataFrame with standardized OHLCV columns and datetime index

        Raises:
            ValueError: If parameters are invalid or no data is returned
            ConnectionError: If API request fails
        """
        # Validate inputs
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")

        self.validate_parameters(symbol, period, interval)

        try:
            # Parse interval into multiplier and timespan
            multiplier, timespan = self._parse_interval(interval)

            # Calculate date range based on period
            end_date = datetime.now().date()
            start_date = self._calculate_start_date(end_date, period)

            # Validate date range
            if start_date >= end_date:
                raise ValueError(
                    f"Invalid date range: start_date {start_date} >= end_date {end_date}"
                )

            # Fetch data from Polygon.io
            aggs = []

            # For minute data, we need to chunk requests to avoid 50k limit
            if timespan == "minute" and (end_date - start_date).days > self._MINUTE_DATA_CHUNK_DAYS:
                # Split into monthly chunks for minute data
                current_date = start_date
                while current_date < end_date:
                    chunk_end = min(
                        current_date + timedelta(days=self._MINUTE_DATA_CHUNK_DAYS), end_date
                    )

                    try:
                        chunk_aggs = list(
                            self.client.get_aggs(
                                ticker=symbol.upper(),  # Ensure uppercase for consistency
                                multiplier=multiplier,
                                timespan=timespan,
                                from_=current_date,
                                to=chunk_end,
                                adjusted=True,
                                sort="asc",
                                limit=50000,
                            )
                        )
                        aggs.extend(chunk_aggs)
                    except Exception as e:
                        raise ConnectionError(
                            f"Failed to fetch chunk {current_date} to {chunk_end}: {e}"
                        ) from e

                    current_date = chunk_end + timedelta(days=1)
            else:
                # Single request for other intervals or short periods
                try:
                    aggs = list(
                        self.client.get_aggs(
                            ticker=symbol.upper(),  # Ensure uppercase for consistency
                            multiplier=multiplier,
                            timespan=timespan,
                            from_=start_date,
                            to=end_date,
                            adjusted=True,
                            sort="asc",
                            limit=50000,
                        )
                    )
                except Exception as e:
                    raise ConnectionError(f"Failed to fetch data for {symbol}: {e}") from e

            if not aggs:
                raise ValueError(
                    f"No data returned for {symbol} in period {period} with interval {interval}"
                )

            # Convert to DataFrame
            df = self._convert_to_dataframe(aggs)

            return self.standardize_dataframe(df)

        except (ValueError, ConnectionError):
            # Re-raise known exceptions without wrapping
            raise
        except Exception as e:
            raise ConnectionError(f"Unexpected error fetching data from Polygon.io: {e}") from e

    def _parse_interval(self, interval: str) -> tuple[int, str]:
        """
        Parse interval string into multiplier and timespan.

        Args:
            interval: Interval string like "15m", "1h", "1d"

        Returns:
            Tuple of (multiplier, timespan)
        """
        # Mapping of common formats to Polygon.io timespan
        interval_mapping = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "30m": (30, "minute"),
            "1h": (1, "hour"),
            "1hour": (1, "hour"),
            "1d": (1, "day"),
            "1day": (1, "day"),
            "daily": (1, "day"),
            "1w": (1, "week"),
            "1wk": (1, "week"),
            "1week": (1, "week"),
            "weekly": (1, "week"),
            "1mo": (1, "month"),
            "1month": (1, "month"),
            "monthly": (1, "month"),
            # Native Polygon.io formats
            "minute": (1, "minute"),
            "hour": (1, "hour"),
            "day": (1, "day"),
            "week": (1, "week"),
            "month": (1, "month"),
        }

        if interval.lower() in interval_mapping:
            return interval_mapping[interval.lower()]

        # Try to parse format like "15min", "5minute", etc.
        match = re.match(r"(\d+)(min|minute|h|hour|d|day|w|week|mo|month)", interval.lower())
        if match:
            multiplier = int(match.group(1))
            unit = match.group(2)

            unit_mapping = {
                "min": "minute",
                "minute": "minute",
                "h": "hour",
                "hour": "hour",
                "d": "day",
                "day": "day",
                "w": "week",
                "week": "week",
                "mo": "month",
                "month": "month",
            }

            timespan = unit_mapping.get(unit, "day")
            return multiplier, timespan

        # Default fallback
        raise ValueError(
            f"Unable to parse interval '{interval}'. Supported formats: 1m, 5m, 15m, 30m, 1h, 1d, etc."
        )

    def _calculate_start_date(self, end_date: date, period: str) -> date:
        """
        Calculate start date based on period string.

        Args:
            end_date: End date for data
            period: Period string like "1y", "6mo", "2mo"

        Returns:
            Start date for data range
        """
        # Period mapping for better maintainability
        period_mapping = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "2mo": 60,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
            "max": 7300,  # ~20 years
        }

        if period == "ytd":
            return datetime(end_date.year, 1, 1).date()

        days_back = period_mapping.get(period, 365)  # Default to 1 year
        return end_date - timedelta(days=days_back)

    def _convert_to_dataframe(self, aggs: list) -> pd.DataFrame:
        """
        Convert Polygon.io aggregates to pandas DataFrame.

        Args:
            aggs: List of aggregate objects from Polygon.io

        Returns:
            DataFrame with OHLCV data and datetime index

        Raises:
            ValueError: If no data provided or data conversion fails
        """
        if not aggs:
            raise ValueError("No aggregates data to convert")

        data = []
        timestamps = []

        for i, agg in enumerate(aggs):
            try:
                # Validate that agg has required attributes
                if (
                    not hasattr(agg, "open")
                    or not hasattr(agg, "high")
                    or not hasattr(agg, "low")
                    or not hasattr(agg, "close")
                    or not hasattr(agg, "volume")
                    or not hasattr(agg, "timestamp")
                ):
                    raise ValueError(f"Aggregate {i} missing required attributes")

                # Convert and validate OHLCV data
                open_price = float(agg.open)
                high_price = float(agg.high)
                low_price = float(agg.low)
                close_price = float(agg.close)
                volume = int(agg.volume)

                # Basic data validation
                if high_price < max(open_price, close_price, low_price):
                    raise ValueError(f"Invalid data: High {high_price} < max(O,C,L) at index {i}")
                if low_price > min(open_price, close_price, high_price):
                    raise ValueError(f"Invalid data: Low {low_price} > min(O,C,H) at index {i}")
                if volume < 0:
                    raise ValueError(f"Invalid data: Negative volume {volume} at index {i}")

                data.append(
                    {
                        "Open": open_price,
                        "High": high_price,
                        "Low": low_price,
                        "Close": close_price,
                        "Volume": volume,
                    }
                )

                # Convert millisecond timestamp to datetime
                timestamps.append(pd.to_datetime(agg.timestamp, unit="ms"))

            except (ValueError, TypeError, AttributeError) as e:
                raise ValueError(f"Error converting aggregate {i}: {e}") from e

        # Create DataFrame with timestamp index
        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))

        # Remove any duplicate timestamps (keep last)
        df = df[~df.index.duplicated(keep="last")]

        return df

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is available through Polygon.io.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            # Try to get recent data for validation
            test_data = list(
                self.client.get_aggs(
                    ticker=symbol.upper(),
                    multiplier=1,
                    timespan="day",
                    from_=datetime.now().date() - timedelta(days=7),
                    to=datetime.now().date(),
                    limit=1,
                )
            )
            return len(test_data) > 0
        except Exception:
            return False
