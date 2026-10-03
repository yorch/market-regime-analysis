"""
Alpaca Markets Data Provider

Historical stock bars from the Alpaca Market Data API v2
(https://docs.alpaca.markets/reference/stockbars).

Authentication uses an API key ID plus a secret key. They can be supplied as:

- ``ProviderConfig(api_key=..., api_secret=...)``
- a single combined ``api_key="KEY_ID:SECRET_KEY"`` (convenient for ``--api-key``)
- the ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` environment variables
  (the names used by Alpaca's own SDKs)

The free Basic plan only serves the IEX feed in real time, so ``feed`` defaults to
``iex``. IEX volume is a small slice of consolidated volume; set ``feed="sip"``
(or ``ALPACA_DATA_FEED=sip``) for consolidated data. On the free plan SIP data must
be at least 15 minutes old, so SIP requests end 16 minutes in the past.

Alpaca intraday bars include pre- and post-market trading. They are filtered to the
regular session (09:30-16:00 America/New_York) by default so intraday features match
other providers; pass ``extended_hours=True`` to keep them.
"""

import os
from datetime import UTC, datetime, time, timedelta
from typing import Any, ClassVar
from urllib.parse import quote

import pandas as pd

from ._http import get_json
from .base import PERIOD_DAYS, MarketDataProvider, ProviderConfig, period_to_start

KEY_ID_ENV = "APCA_API_KEY_ID"
SECRET_KEY_ENV = "APCA_API_SECRET_KEY"
FEED_ENV = "ALPACA_DATA_FEED"


class AlpacaProvider(MarketDataProvider):
    """Alpaca Markets data provider (key ID + secret key authentication)."""

    provider_name = "alpaca"

    BASE_URL = "https://data.alpaca.markets/v2/stocks"
    _PAGE_LIMIT = 10000  # Maximum bars per page allowed by the API
    _MAX_PAGES = 100  # Safety cap against runaway pagination
    _SIP_DELAY = timedelta(minutes=16)  # Free plan: SIP data must be >15 min old
    _FEEDS: ClassVar[set[str]] = {"iex", "sip", "delayed_sip", "boats", "otc"}
    _MARKET_TZ = "America/New_York"
    _SESSION_OPEN = time(9, 30)
    _SESSION_CLOSE = time(16, 0)
    _SATURDAY = 5  # pandas dayofweek: Monday=0

    _INTERVAL_MAP: ClassVar[dict[str, str]] = {
        "1m": "1Min",
        "1min": "1Min",
        "5m": "5Min",
        "5min": "5Min",
        "15m": "15Min",
        "15min": "15Min",
        "30m": "30Min",
        "30min": "30Min",
        "1h": "1Hour",
        "1hour": "1Hour",
        "60min": "1Hour",
        "1d": "1Day",
        "1day": "1Day",
        "daily": "1Day",
        "1w": "1Week",
        "1wk": "1Week",
        "1week": "1Week",
        "weekly": "1Week",
        "1mo": "1Month",
        "1month": "1Month",
        "monthly": "1Month",
    }

    supported_intervals: ClassVar[set[str]] = set(_INTERVAL_MAP)
    supported_periods: ClassVar[set[str]] = {*PERIOD_DAYS, "ytd"}
    requires_api_key = True
    rate_limit_per_minute = 200  # Basic (free) plan
    description = "Alpaca Markets stock bars (IEX feed on the free plan, SIP with subscription)"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        """Initialize Alpaca provider, resolving credentials from config or environment."""
        super().__init__(config)

        feed = getattr(self.config, "feed", None) or os.getenv(FEED_ENV) or "iex"
        if feed not in self._FEEDS:
            raise ValueError(f"Unsupported Alpaca feed '{feed}'. Supported: {sorted(self._FEEDS)}")
        self.feed: str = feed
        self.extended_hours: bool = bool(getattr(self.config, "extended_hours", False))

    def _validate_config(self) -> None:
        """Resolve the key ID and secret key; both are required, from the same source."""
        key_id = self.config.api_key
        secret = getattr(self.config, "api_secret", None)

        if key_id:
            # Accept "KEY_ID:SECRET" so a single --api-key flag can carry both. An
            # explicit key ID is never paired with a secret from the environment.
            if not secret and ":" in key_id:
                key_id, secret = key_id.split(":", 1)
        else:
            key_id, secret = os.getenv(KEY_ID_ENV), os.getenv(SECRET_KEY_ENV)

        if not key_id or not secret:
            raise ValueError(
                f"Provider 'alpaca' requires an API key ID and secret key. Set {KEY_ID_ENV} "
                f"and {SECRET_KEY_ENV}, or pass api_key='KEY_ID:SECRET_KEY'."
            )

        self._key_id: str = key_id
        self._secret: str = secret

    @property
    def _headers(self) -> dict[str, str]:
        return {"APCA-API-KEY-ID": self._key_id, "APCA-API-SECRET-KEY": self._secret}

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetch historical bars from Alpaca.

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

        end = datetime.now(UTC)
        if self.feed == "sip":
            end -= self._SIP_DELAY
        start = period_to_start(period, end)

        params: dict[str, Any] = {
            "timeframe": self._INTERVAL_MAP[interval],
            "start": start.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "end": end.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "limit": self._PAGE_LIMIT,
            "adjustment": "all",
            "feed": self.feed,
            "sort": "asc",
        }

        bars: list[dict[str, Any]] = []
        url = f"{self.BASE_URL}/{quote(symbol.upper(), safe='')}/bars"
        for _ in range(self._MAX_PAGES):
            payload = get_json(
                url,
                provider="Alpaca",
                config=self.config,
                params=params,
                headers=self._headers,
                throttle=self.throttle,
            )
            bars.extend(payload.get("bars") or [])
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        else:
            raise ConnectionError(
                f"Alpaca pagination exceeded {self._MAX_PAGES} pages for {symbol}; "
                "use a shorter period or a coarser interval"
            )

        if not bars:
            raise ValueError(
                f"No data returned for {symbol} in period {period} with interval {interval}"
            )

        df = self._bars_to_dataframe(bars)
        timeframe = params["timeframe"]
        if not self.extended_hours and timeframe.endswith(("Min", "Hour")):
            df = self._regular_session(df, timeframe)
            if df.empty:
                raise ValueError(f"No regular-session data returned for {symbol}")

        return self.standardize_dataframe(df, interval)

    @classmethod
    def _regular_session(cls, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Keep weekday bars that overlap the regular session (e.g. the 09:00 hourly bar)."""
        unit = "min" if timeframe.endswith("Min") else "h"
        width = pd.Timedelta(int(timeframe.removesuffix("Min").removesuffix("Hour")), unit=unit)

        local = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert(cls._MARKET_TZ)
        clock = local.to_series().dt
        minutes = (clock.hour * 60 + clock.minute).to_numpy()
        open_min = cls._SESSION_OPEN.hour * 60 + cls._SESSION_OPEN.minute
        close_min = cls._SESSION_CLOSE.hour * 60 + cls._SESSION_CLOSE.minute
        width_min = int(width.total_seconds() // 60)

        overlaps = (minutes < close_min) & (minutes + width_min > open_min)
        session: pd.DataFrame = df[overlaps & (clock.dayofweek < cls._SATURDAY).to_numpy()]
        return session

    @staticmethod
    def _bars_to_dataframe(bars: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert Alpaca bar objects (t/o/h/l/c/v keys) into an OHLCV DataFrame."""
        try:
            df = pd.DataFrame(
                {
                    "Open": [float(b["o"]) for b in bars],
                    "High": [float(b["h"]) for b in bars],
                    "Low": [float(b["l"]) for b in bars],
                    "Close": [float(b["c"]) for b in bars],
                    "Volume": [int(b["v"]) for b in bars],
                },
                index=pd.to_datetime([b["t"] for b in bars], utc=True).tz_convert(None),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Malformed bar data from Alpaca: {e}") from e

        deduped: pd.DataFrame = df[~df.index.duplicated(keep="last")]
        return deduped
