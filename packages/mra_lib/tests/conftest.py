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
