"""
Shared, causal feature engineering for regime detection and technical analysis.

Every function here is *pure* (no hidden state, inputs are never mutated) and
*causal*: the value at bar ``t`` depends only on bars ``<= t``. Truncating or
changing future bars therefore never changes past feature values, which is
what makes walk-forward validation and per-bar regime history leak-free.

:func:`build_hmm_features` assembles the small, stationary feature set the
HMM regime detector is fitted on. The other helpers are reused by
:class:`~mra_lib.analyzer.MarketRegimeAnalyzer` for its display indicators so
the formulas exist exactly once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Small constant guarding logs of zero volatility.
_LOG_EPS = 1e-12

#: Rolling windows (bars) used by :func:`build_hmm_features`.
VOL_WINDOW = 20
VOL_SHORT_WINDOW = 10
VOL_LONG_WINDOW = 40
TREND_FAST_WINDOW = 10
TREND_SLOW_WINDOW = 30
AUTOCORR_WINDOW = 30
VOLUME_WINDOW = 20

#: Columns produced by :func:`build_hmm_features`, in order.
HMM_FEATURES: tuple[str, ...] = (
    "log_return",
    "log_volatility",
    "vol_expansion",
    "trend_strength",
    "autocorr_1",
    "log_volume_ratio",
)

#: Bars of history consumed before the first complete HMM feature row.
HMM_WARMUP_BARS = max(
    VOL_LONG_WINDOW,
    TREND_SLOW_WINDOW,
    AUTOCORR_WINDOW + 1,
    VOL_WINDOW + 1,
    VOLUME_WINDOW,
)


def log_returns(close: pd.Series) -> pd.Series:
    """Bar-to-bar log return ``log(C_t / C_{t-1})`` (NaN on the first bar)."""
    return pd.Series(np.log(close / close.shift(1)), index=close.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    """
    True range: ``max(H - L, |H - C_prev|, |L - C_prev|)``.

    The first bar has no previous close and falls back to ``H - L``.
    """
    prev_close = df["Close"].shift(1)
    ranges = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True)


def average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Simple moving average of :func:`true_range` over ``window`` bars (price units)."""
    result: pd.Series = true_range(df).rolling(window).mean()
    return result


def normalized_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR divided by the close: a scale-free range measure (fraction of price)."""
    result: pd.Series = average_true_range(df, window) / df["Close"]
    return result


def rolling_volatility(returns: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Rolling sample standard deviation of ``returns`` (per-bar, not annualized)."""
    result: pd.Series = returns.rolling(window).std()
    return result


def rolling_log_volatility(returns: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Natural log of :func:`rolling_volatility` (near-Gaussian, stationary)."""
    return pd.Series(np.log(rolling_volatility(returns, window) + _LOG_EPS), index=returns.index)


def rolling_autocorr(series: pd.Series, lag: int = 1, window: int = AUTOCORR_WINDOW) -> pd.Series:
    """
    Vectorized rolling lag-``lag`` autocorrelation.

    Pearson correlation of the pairs ``(x_s, x_{s-lag})`` for the last
    ``window`` values of ``s``. A window with zero variance has no defined
    correlation and returns ``0.0``; the result is clipped to ``[-1, 1]``.
    """
    lagged = series.shift(lag)
    roll = series.rolling(window)
    cov = roll.cov(lagged)
    var_x = roll.var()
    var_y = lagged.rolling(window).var()
    denom = np.sqrt(var_x * var_y)
    corr = cov / denom.where(denom > 0)
    corr = corr.where(~(denom.notna() & (denom <= 0)), 0.0)
    result: pd.Series = corr.clip(-1.0, 1.0)
    return result


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    ``(x_t - mean) / std`` over the trailing ``window`` bars (current bar included).

    A window with zero standard deviation returns ``0.0``.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / std.where(std > 0)
    result: pd.Series = z.where(~(std.notna() & (std <= 0)), 0.0)
    return result


def trend_strength(
    close: pd.Series, fast: int = TREND_FAST_WINDOW, slow: int = TREND_SLOW_WINDOW
) -> pd.Series:
    """
    Scale-free trend: ``(SMA_fast - SMA_slow) / close``.

    For a constant per-bar drift ``mu`` this is approximately
    ``mu * (slow - fast) / 2``.
    """
    result: pd.Series = (close.rolling(fast).mean() - close.rolling(slow).mean()) / close
    return result


def volume_ratio(volume: pd.Series, window: int = VOLUME_WINDOW) -> pd.Series:
    """
    ``volume / rolling-mean(volume)``, decided per bar.

    Bars whose current volume or trailing mean volume is not positive (e.g.
    FX or index data without volume) get a neutral ``1.0``. The rule only
    looks at the current window, so it is causal; it replaces a whole-series
    ``Volume.sum() > 0`` test that let future bars change past features.
    """
    vol = volume.astype(float)
    mean = vol.rolling(window).mean()
    valid = (vol > 0) & (mean > 0)
    ratio = (vol / mean.where(mean > 0)).where(valid, 1.0)
    result: pd.Series = ratio.where(mean.notna())
    return result


def build_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the stationary feature matrix the HMM regime detector is fitted on.

    Six low-collinearity, scale-free features (columns in :data:`HMM_FEATURES`):

    ``log_return``
        Bar log return.
    ``log_volatility``
        Log of the 20-bar standard deviation of log returns.
    ``vol_expansion``
        ``log(vol_10 / vol_40)``: positive when volatility is expanding.
    ``trend_strength``
        ``(SMA_10 - SMA_30) / close``.
    ``autocorr_1``
        30-bar lag-1 autocorrelation of log returns.
    ``log_volume_ratio``
        Log of volume over its 20-bar mean (``0`` when volume is unavailable).

    No raw price levels, moving averages, ATR or price changes are included,
    so the features are invariant to rescaling prices. All features are
    causal; the first :data:`HMM_WARMUP_BARS` warm-up rows are dropped.

    Args:
        df: OHLCV frame with at least a ``Close`` column (``Volume`` optional)

    Returns:
        Feature frame indexed like ``df`` (warm-up rows removed), no NaN/inf

    Raises:
        ValueError: If ``df`` has too few bars to produce any feature row
    """
    if "Close" not in df.columns:
        raise ValueError("OHLCV data must contain a 'Close' column")
    if len(df) <= HMM_WARMUP_BARS:
        raise ValueError(
            f"Insufficient data for feature calculation "
            f"(need more than {HMM_WARMUP_BARS} bars, got {len(df)})"
        )

    close = df["Close"].astype(float)
    rets = log_returns(close)

    features = pd.DataFrame(index=df.index)
    features["log_return"] = rets
    features["log_volatility"] = rolling_log_volatility(rets, VOL_WINDOW)
    features["vol_expansion"] = rolling_log_volatility(
        rets, VOL_SHORT_WINDOW
    ) - rolling_log_volatility(rets, VOL_LONG_WINDOW)
    features["trend_strength"] = trend_strength(close)
    features["autocorr_1"] = rolling_autocorr(rets, lag=1, window=AUTOCORR_WINDOW)
    if "Volume" in df.columns:
        features["log_volume_ratio"] = np.log(volume_ratio(df["Volume"], VOLUME_WINDOW))
    else:
        features["log_volume_ratio"] = pd.Series(0.0, index=df.index).where(
            close.rolling(VOLUME_WINDOW).count() == VOLUME_WINDOW
        )

    features = features.replace([np.inf, -np.inf], np.nan).iloc[HMM_WARMUP_BARS:]
    clean: pd.DataFrame = features.dropna()
    return clean
