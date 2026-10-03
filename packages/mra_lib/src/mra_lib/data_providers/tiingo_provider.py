"""
Tiingo Data Provider

End-of-day prices from the Tiingo daily endpoint (split/dividend adjusted) and
intraday bars from the Tiingo IEX endpoint (https://www.tiingo.com/documentation).

The API key is read from ``ProviderConfig(api_key=...)`` or ``TIINGO_API_KEY`` and
sent in the ``Authorization`` header rather than the query string.
"""

import os
from datetime import UTC, datetime
from typing import Any, ClassVar
from urllib.parse import quote

import pandas as pd

from ._http import get_json
from .base import (
    PERIOD_DAYS,
    InvalidSymbolError,
    MarketDataProvider,
    ProviderConfig,
    period_to_start,
)

API_KEY_ENV = "TIINGO_API_KEY"


class TiingoProvider(MarketDataProvider):
    """Tiingo data provider with API token authentication."""

    provider_name = "tiingo"

    DAILY_URL = "https://api.tiingo.com/tiingo/daily"
    IEX_URL = "https://api.tiingo.com/iex"

    # interval -> (endpoint kind, Tiingo resampleFreq)
    _INTERVAL_MAP: ClassVar[dict[str, tuple[str, str]]] = {
        "1m": ("iex", "1min"),
        "1min": ("iex", "1min"),
        "5m": ("iex", "5min"),
        "5min": ("iex", "5min"),
        "15m": ("iex", "15min"),
        "15min": ("iex", "15min"),
        "30m": ("iex", "30min"),
        "30min": ("iex", "30min"),
        "1h": ("iex", "1hour"),
        "1hour": ("iex", "1hour"),
        "60min": ("iex", "1hour"),
        "1d": ("daily", "daily"),
        "1day": ("daily", "daily"),
        "daily": ("daily", "daily"),
        "1w": ("daily", "weekly"),
        "1wk": ("daily", "weekly"),
        "1week": ("daily", "weekly"),
        "weekly": ("daily", "weekly"),
        "1mo": ("daily", "monthly"),
        "1month": ("daily", "monthly"),
        "monthly": ("daily", "monthly"),
    }

    supported_intervals: ClassVar[set[str]] = set(_INTERVAL_MAP)
    supported_periods: ClassVar[set[str]] = {*PERIOD_DAYS, "ytd"}
    requires_api_key = True
    rate_limit_per_minute = 1  # Free tier: 50 requests/hour, 1000/day
    rate_limit_burst = 10  # Allow a short burst (e.g. all analyzer timeframes) before spacing
    description = "Tiingo adjusted end-of-day prices and IEX intraday bars"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize Tiingo provider, falling back to TIINGO_API_KEY."""
        config = config or ProviderConfig()
        if not config.api_key:
            config.api_key = os.getenv(API_KEY_ENV)
        super().__init__(config)

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetch price data from Tiingo.

        Args:
            symbol: Trading symbol (e.g., 'SPY', 'AAPL')
            period: Time period for data (e.g., '1y', '2y', '6mo')
            interval: Data interval (e.g., '1d', '1h', '15m')

        Returns:
            DataFrame following the ``base`` contract (tz-naive UTC intraday index,
            session dates for daily bars)

        Raises:
            ValueError: If parameters are invalid or no data is returned
            ConnectionError: If the API request fails
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")

        interval = interval.lower()
        self.validate_parameters(symbol, period, interval)

        kind, freq = self._INTERVAL_MAP[interval]
        end = datetime.now(UTC)
        start = period_to_start(period, end)
        ticker = quote(symbol.lower(), safe="")

        params: dict[str, Any] = {
            "startDate": start.date().isoformat(),
            "endDate": end.date().isoformat(),
            "resampleFreq": freq,
            "format": "json",
        }
        if kind == "daily":
            url = f"{self.DAILY_URL}/{ticker}/prices"
        else:
            url = f"{self.IEX_URL}/{ticker}/prices"
            # Volume is only returned for IEX bars when requested explicitly
            params["columns"] = "open,high,low,close,volume"

        rows = get_json(
            url,
            provider="Tiingo",
            config=self.config,
            params=params,
            headers={"Authorization": f"Token {self.config.api_key}"},
            throttle=self.throttle,
        )

        # Tiingo reports errors (unknown ticker, exhausted quota) as {"detail": ...}
        if isinstance(rows, dict) and rows.get("detail"):
            detail = str(rows["detail"])
            if "not found" in detail.lower():
                raise InvalidSymbolError(f"Tiingo error for {symbol}: {detail}")
            raise ConnectionError(f"Tiingo error for {symbol}: {detail}")
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"No data returned for {symbol} in period {period} with interval {interval}"
            )

        return self.standardize_dataframe(
            self._rows_to_dataframe(rows, adjusted=kind == "daily"), interval
        )

    @staticmethod
    def _rows_to_dataframe(rows: list[dict[str, Any]], *, adjusted: bool) -> pd.DataFrame:
        """Convert Tiingo price rows into an OHLCV DataFrame."""
        prefix = "adj" if adjusted else ""

        def col(name: str) -> str:
            return f"{prefix}{name.capitalize()}" if prefix else name

        try:
            df = pd.DataFrame(
                {
                    "Open": [float(r[col("open")]) for r in rows],
                    "High": [float(r[col("high")]) for r in rows],
                    "Low": [float(r[col("low")]) for r in rows],
                    "Close": [float(r[col("close")]) for r in rows],
                    "Volume": [int(r[col("volume")] or 0) for r in rows],
                },
                index=pd.to_datetime([r["date"] for r in rows], utc=True).tz_convert(None),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Malformed price data from Tiingo: {e}") from e

        deduped: pd.DataFrame = df[~df.index.duplicated(keep="last")]
        return deduped
