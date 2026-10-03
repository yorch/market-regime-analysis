"""
Polygon.io Data Provider

Aggregate bars from Polygon.io via the official ``polygon-api-client``.
``list_aggs`` follows ``next_url`` pagination, so long minute-level ranges are
not silently truncated at the per-request limit. Timeouts and retries come from
``ProviderConfig.timeout`` / ``ProviderConfig.retries``.
"""

import logging
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar

import pandas as pd

from .base import (
    PERIOD_DAYS,
    AuthError,
    InvalidSymbolError,
    MarketDataProvider,
    ProviderConfig,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Units accepted in "<n><unit>" interval strings (e.g. "15min", "2hour")
_UNIT_MAP = {
    "m": "minute",
    "min": "minute",
    "minute": "minute",
    "h": "hour",
    "hour": "hour",
    "d": "day",
    "day": "day",
    "w": "week",
    "wk": "week",
    "week": "week",
    "mo": "month",
    "month": "month",
    "q": "quarter",
    "quarter": "quarter",
    "y": "year",
    "year": "year",
}
_INTERVAL_PATTERN = re.compile(r"(\d+)(" + "|".join(sorted(_UNIT_MAP, key=len, reverse=True)) + ")")


class PolygonProvider(MarketDataProvider):
    """Polygon.io data provider with API key authentication."""

    provider_name = "polygon"

    # Fixed interval spellings -> (multiplier, timespan)
    _INTERVAL_MAP: ClassVar[dict[str, tuple[int, str]]] = {
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
        "minute": (1, "minute"),
        "hour": (1, "hour"),
        "day": (1, "day"),
        "week": (1, "week"),
        "month": (1, "month"),
        "quarter": (1, "quarter"),
        "year": (1, "year"),
    }

    supported_intervals: ClassVar[set[str]] = set(_INTERVAL_MAP)
    supported_periods: ClassVar[set[str]] = {*PERIOD_DAYS, "ytd"}
    requires_api_key = True
    rate_limit_per_minute = 5  # Free (Basic) plan; paid plans are unlimited
    description = "Polygon.io aggregates (free plan: 5 requests/minute, 2 years history)"

    _PAGE_LIMIT = 50000  # Maximum bars per page allowed by the API

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize Polygon.io provider with API key."""
        super().__init__(config)

        try:
            from polygon import RESTClient  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "polygon-api-client library is required. "
                "Install with: pip install polygon-api-client"
            ) from e

        timeout = float(self.config.timeout)
        self.client: Any = RESTClient(
            api_key=self.config.api_key,
            connect_timeout=timeout,
            read_timeout=timeout,
            retries=self.config.retries,
        )

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetch data from Polygon.io API.

        Args:
            symbol: Trading symbol (e.g., 'SPY', 'AAPL')
            period: Time period for data (e.g., '1y', '2y', '6mo')
            interval: Data interval (e.g., '1d', '1h', '15m')

        Returns:
            DataFrame following the ``base`` contract

        Raises:
            ValueError: If parameters are invalid
            InvalidSymbolError: If no data is returned
            AuthError / RateLimitError / ConnectionError: If the API request fails
        """
        interval = interval.lower()
        self.validate_parameters(symbol, period, interval)
        multiplier, timespan = self._parse_interval(interval)

        end_date = datetime.now(UTC).date()
        start_date = self._calculate_start_date(end_date, period)
        if start_date >= end_date:
            raise ValueError(f"Invalid date range: start_date {start_date} >= end_date {end_date}")

        self.throttle()
        try:
            aggs = list(
                self.client.list_aggs(
                    ticker=symbol.upper(),
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=start_date,
                    to=end_date,
                    adjusted=True,
                    sort="asc",
                    limit=self._PAGE_LIMIT,
                )
            )
        except Exception as e:
            raise self._classify(symbol, e) from e

        if not aggs:
            raise InvalidSymbolError(
                f"No data returned for {symbol} in period {period} with interval {interval}"
            )

        df = self._convert_to_dataframe(aggs)
        if df.empty:
            raise InvalidSymbolError(f"No valid bars returned for {symbol}")
        return self.standardize_dataframe(df, interval)

    @staticmethod
    def _classify(symbol: str, error: Exception) -> Exception:
        """Map polygon client exceptions onto the provider error hierarchy."""
        name = type(error).__name__
        message = str(error)
        lowered = message.lower()
        if name == "AuthError" or "not_authorized" in lowered or "api key" in lowered:
            return AuthError(f"Polygon.io rejected the API key: {message[:200]}")
        # Note: urllib3's "Max retries exceeded" also appears for DNS/connection errors
        if (
            "exceeded the maximum requests" in lowered
            or "too many 429" in lowered
            or "429" in lowered
        ):
            return RateLimitError(f"Polygon.io rate limit reached: {message[:200]}")
        if isinstance(error, ValueError):
            return error
        return ConnectionError(
            f"Failed to fetch data for {symbol} from Polygon.io: {message[:200]}"
        )

    def _parse_interval(self, interval: str) -> tuple[int, str]:
        """
        Parse interval string into multiplier and timespan.

        Args:
            interval: Interval string like "15m", "1h", "1d", "15min"

        Returns:
            Tuple of (multiplier, timespan)
        """
        key = interval.lower()
        if key in self._INTERVAL_MAP:
            return self._INTERVAL_MAP[key]

        match = _INTERVAL_PATTERN.fullmatch(key)
        if match:
            return int(match.group(1)), _UNIT_MAP[match.group(2)]

        raise ValueError(
            f"Unable to parse interval '{interval}'. Supported formats: 1m, 5m, 15m, 30m, 1h, 1d, etc."
        )

    def _calculate_start_date(self, end_date: date, period: str) -> date:
        """
        Calculate start date based on period string.

        Args:
            end_date: End date for data
            period: Period string like "1y", "6mo", "ytd"

        Returns:
            Start date for data range
        """
        if period == "ytd":
            return date(end_date.year, 1, 1)
        if period not in PERIOD_DAYS:
            raise ValueError(f"Unsupported period '{period}'. Supported: {sorted(PERIOD_DAYS)}")
        return end_date - timedelta(days=PERIOD_DAYS[period])

    def _convert_to_dataframe(self, aggs: list) -> pd.DataFrame:
        """
        Convert Polygon.io aggregates to a DataFrame, dropping malformed bars.

        A single bad bar (missing fields, High < Low, negative volume) is dropped
        with a warning instead of failing the whole fetch.

        Args:
            aggs: List of aggregate objects from Polygon.io

        Returns:
            DataFrame with OHLCV data and a naive-UTC datetime index

        Raises:
            ValueError: If no aggregates are provided
        """
        if not aggs:
            raise ValueError("No aggregates data to convert")

        data = []
        timestamps = []
        dropped = 0

        for agg in aggs:
            try:
                open_price = float(agg.open)
                high_price = float(agg.high)
                low_price = float(agg.low)
                close_price = float(agg.close)
                volume = float(agg.volume)
                timestamp = pd.to_datetime(int(agg.timestamp), unit="ms")
            except (TypeError, ValueError, AttributeError):
                dropped += 1
                continue

            if (
                high_price < max(open_price, close_price, low_price)
                or low_price > min(open_price, close_price, high_price)
                or volume < 0
            ):
                dropped += 1
                continue

            data.append(
                {
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume,
                }
            )
            timestamps.append(timestamp)

        if dropped:
            logger.warning("Polygon.io: dropped %d malformed bar(s) of %d", dropped, len(aggs))

        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))
        deduped: pd.DataFrame = df[~df.index.duplicated(keep="last")]
        return deduped

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is available through Polygon.io.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if symbol is valid, False otherwise
        """
        today = datetime.now(UTC).date()
        try:
            self.throttle()
            test_data = list(
                self.client.get_aggs(
                    ticker=symbol.upper(),
                    multiplier=1,
                    timespan="day",
                    from_=today - timedelta(days=7),
                    to=today,
                    limit=1,
                )
            )
            return len(test_data) > 0
        except Exception:  # noqa: BLE001 - a bool probe: any client failure means "not valid"
            return False
