"""Shared pytest fixtures for mra_lib tests."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest


def make_synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Build deterministic synthetic OHLCV data with an up-trend then a down-trend.

    Parameters
    ----------
    n : int
        Number of business-day bars.
    seed : int
        Seed for the random generator, so results are reproducible.

    Returns
    -------
    pd.DataFrame
        Frame with ``Open``, ``High``, ``Low``, ``Close`` and ``Volume`` columns.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    log_prices = np.concatenate(
        [
            np.cumsum(rng.normal(0.001, 0.01, half)),
            np.cumsum(rng.normal(-0.001, 0.015, n - half)) + 0.001 * half,
        ]
    )
    close = 100 * np.exp(log_prices)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(500_000, 2_000_000, n),
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )


def make_switching_ohlcv(n: int = 600, seed: int = 3) -> pd.DataFrame:
    """
    Deterministic OHLCV data cycling through calm-bull, volatile-bear and choppy regimes.

    Each regime lasts 60-140 bars; intrabar ranges scale with the regime's volatility.
    """
    rng = np.random.default_rng(seed)
    regimes = [(0.0008, 0.007), (-0.001, 0.022), (0.0, 0.012)]
    rets: list[float] = []
    vols: list[float] = []
    k = 0
    while len(rets) < n:
        mu, sd = regimes[k % 3]
        length = int(rng.integers(60, 140))
        rets.extend(rng.normal(mu, sd, length))
        vols.extend([sd] * length)
        k += 1
    r = np.array(rets[:n])
    sd_arr = np.array(vols[:n])
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    spread = np.abs(rng.normal(0, 1, n)) * sd_arr * 0.4
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * (1 + spread),
            "Low": np.minimum(open_, close) * (1 - spread),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=pd.bdate_range("2018-01-01", periods=n),
    )


@pytest.fixture(scope="session")
def switching_ohlcv() -> Callable[..., pd.DataFrame]:
    """Return the :func:`make_switching_ohlcv` factory."""
    return make_switching_ohlcv


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> Callable[..., pd.DataFrame]:
    """Return the :func:`make_synthetic_ohlcv` factory."""
    return make_synthetic_ohlcv


@pytest.fixture(autouse=True)
def _reset_provider_rate_limiters():
    """Provider rate-limit buckets are shared per process; isolate tests from each other."""
    from mra_lib.data_providers import MarketDataProvider

    MarketDataProvider.reset_rate_limiters()
    yield
    MarketDataProvider.reset_rate_limiters()
