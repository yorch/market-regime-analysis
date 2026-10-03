"""Shared feature module: causality, stationarity (scale invariance), and NaN hygiene."""

import numpy as np
import pandas as pd
import pytest

from mra_lib.indicators import features as F
from mra_lib.indicators.features import HMM_FEATURES, HMM_WARMUP_BARS, build_hmm_features


@pytest.fixture
def ohlcv(synthetic_ohlcv) -> pd.DataFrame:
    return synthetic_ohlcv(n=300)


def _perturb_future(df: pd.DataFrame, cut: int, seed: int = 1) -> pd.DataFrame:
    """Replace every bar from ``cut`` on with different data."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(df) - cut
    factor = np.exp(np.cumsum(rng.normal(0.0, 0.03, n)))
    for col in ("Open", "High", "Low", "Close"):
        out.iloc[cut:, out.columns.get_loc(col)] = df[col].to_numpy()[cut:] * factor * 1.5
    out.iloc[cut:, out.columns.get_loc("Volume")] = rng.integers(1, 10, n)
    return out


class TestBuildHmmFeatures:
    def test_columns_and_size(self, ohlcv):
        X = build_hmm_features(ohlcv)
        assert list(X.columns) == list(HMM_FEATURES)
        assert 5 <= X.shape[1] <= 7
        assert len(X) == len(ohlcv) - HMM_WARMUP_BARS

    def test_no_nan_or_inf(self, ohlcv):
        X = build_hmm_features(ohlcv)
        assert np.isfinite(X.to_numpy()).all()

    def test_insufficient_data(self, ohlcv):
        with pytest.raises(ValueError, match="Insufficient data"):
            build_hmm_features(ohlcv.iloc[:HMM_WARMUP_BARS])

    @pytest.mark.parametrize("cut", [HMM_WARMUP_BARS + 5, 150, 299])
    def test_truncation_invariance(self, ohlcv, cut):
        """Features up to bar t are unchanged when bars after t change or disappear."""
        base = build_hmm_features(ohlcv)
        prefix = build_hmm_features(ohlcv.iloc[:cut])
        changed = build_hmm_features(_perturb_future(ohlcv, cut))
        past = base.index[base.index < ohlcv.index[cut]]
        pd.testing.assert_frame_equal(prefix.loc[past], base.loc[past])
        pd.testing.assert_frame_equal(changed.loc[past], base.loc[past])

    def test_price_scale_invariance(self, ohlcv):
        scaled = ohlcv.copy()
        scaled[["Open", "High", "Low", "Close"]] *= 10.0
        scaled["Volume"] *= 3
        pd.testing.assert_frame_equal(
            build_hmm_features(scaled), build_hmm_features(ohlcv), atol=1e-9, rtol=1e-7
        )

    def test_low_collinearity(self, ohlcv):
        corr = build_hmm_features(ohlcv).corr().to_numpy()
        off_diag = np.abs(corr[~np.eye(len(corr), dtype=bool)])
        assert off_diag.max() < 0.6

    def test_volume_missing_or_zero_is_neutral(self, ohlcv):
        no_volume = build_hmm_features(ohlcv.drop(columns="Volume"))
        zero_volume = build_hmm_features(ohlcv.assign(Volume=0))
        assert (no_volume["log_volume_ratio"] == 0).all()
        assert (zero_volume["log_volume_ratio"] == 0).all()

    def test_volume_rule_is_per_bar(self, ohlcv):
        """Volume appearing later must not change earlier rows (old whole-prefix rule did)."""
        df = ohlcv.copy()
        df.iloc[:200, df.columns.get_loc("Volume")] = 0
        full = build_hmm_features(df)
        prefix = build_hmm_features(df.iloc[:200])
        pd.testing.assert_frame_equal(full.loc[prefix.index], prefix)


class TestHelpers:
    def test_true_range_first_bar_is_high_low(self, ohlcv):
        tr = F.true_range(ohlcv)
        assert tr.iloc[0] == pytest.approx(ohlcv["High"].iloc[0] - ohlcv["Low"].iloc[0])
        assert (tr >= ohlcv["High"] - ohlcv["Low"] - 1e-12).all()

    def test_normalized_atr_is_scale_free(self, ohlcv):
        scaled = ohlcv.copy()
        scaled[["Open", "High", "Low", "Close"]] *= 7
        pd.testing.assert_series_equal(F.normalized_atr(scaled), F.normalized_atr(ohlcv))

    def test_rolling_autocorr_matches_pandas_corr(self, ohlcv):
        r = F.log_returns(ohlcv["Close"])
        fast = F.rolling_autocorr(r, lag=1, window=30)
        t = 120
        window = r.iloc[t - 29 : t + 1]
        lagged = r.shift(1).iloc[t - 29 : t + 1]
        assert fast.iloc[t] == pytest.approx(window.corr(lagged))

    def test_rolling_autocorr_constant_window_is_zero(self):
        s = pd.Series([0.01] * 50)
        assert (F.rolling_autocorr(s, window=10).dropna() == 0).all()

    def test_rolling_zscore_constant_is_zero(self):
        s = pd.Series([5.0] * 30)
        assert (F.rolling_zscore(s, 10).dropna() == 0).all()

    def test_trend_strength_sign(self):
        up = pd.Series(np.exp(np.linspace(0, 1, 100)))
        assert (F.trend_strength(up).dropna() > 0).all()
        assert (F.trend_strength(up[::-1].reset_index(drop=True)).dropna() < 0).all()
