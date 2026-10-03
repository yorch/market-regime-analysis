"""
Hidden Markov Model regime detector implementation.

This module implements the core HMM functionality following Jim Simons'
mathematical approach for market regime detection.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from mra_lib.config.enums import MarketRegime

# Regime-classification thresholds. Features are standardized (z-units) before
# clustering, so these are expressed in standard deviations of each feature.
_TREND_Z = 0.2  # |mean z| of trend_strength needed to call a trend
_BREAKOUT_Z = 0.5  # |mean z| of returns needed to call a breakout
_RECENT_WINDOW = 20  # bars used to characterise the current state


class HiddenMarkovRegimeDetector:
    """
    True HMM implementation following Simons' mathematical approach.

    This class implements Hidden Markov Models for market regime detection
    using Gaussian Mixture Models as the emission distributions and proper
    transition matrix estimation.

    The implementation follows Renaissance Technologies' approach with:
    - Multi-feature mathematical analysis
    - Higher-order moments (skewness, kurtosis)
    - Cross-correlations between features
    - Proper transition matrix estimation
    - Regime persistence metrics
    """

    def __init__(self, n_states: int = 6) -> None:
        """
        Initialize the HMM detector.

        Args:
            n_states: Number of hidden states (default 6 for comprehensive regime detection)
        """
        self.n_states = n_states
        self.gmm: GaussianMixture | None = None
        self.scaler: StandardScaler | None = None
        self.transition_matrix: np.ndarray | None = None
        self.state_means: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.fitted: bool = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive mathematical features for HMM analysis.

        This method implements the sophisticated feature engineering
        used by Renaissance Technologies, including higher-order moments
        and cross-correlations.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features

        Raises:
            ValueError: If insufficient data for feature calculation
        """
        if len(df) < 50:
            raise ValueError("Insufficient data for feature calculation (minimum 50 bars)")

        features = pd.DataFrame(index=df.index)

        # Basic price features
        features["returns"] = df["Close"].pct_change()
        features["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        features["price_change"] = df["Close"] - df["Close"].shift(1)

        # Volatility features
        features["volatility"] = features["returns"].rolling(20).std()
        features["log_volatility"] = np.log(features["volatility"] + 1e-8)

        # ATR calculation
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift(1))
        low_close = np.abs(df["Low"] - df["Close"].shift(1))
        true_range = pd.Series(
            np.maximum(high_low, np.maximum(high_close, low_close)), index=df.index
        )
        features["atr"] = true_range.rolling(14).mean()
        features["atr_normalized"] = features["atr"] / df["Close"]

        # Higher-order moments (Simons signature)
        window = 20
        features["skewness"] = features["returns"].rolling(window).skew()
        features["kurtosis"] = features["returns"].rolling(window).kurt()

        # Trend strength features
        features["sma_9"] = df["Close"].rolling(9).mean()
        features["sma_21"] = df["Close"].rolling(21).mean()
        features["sma_50"] = df["Close"].rolling(50).mean()

        features["trend_strength"] = (features["sma_9"] - features["sma_21"]) / df["Close"]
        features["long_trend"] = (features["sma_21"] - features["sma_50"]) / df["Close"]

        # Autocorrelation features (momentum persistence)
        for lag in [1, 2, 5]:
            features[f"autocorr_{lag}"] = (
                features["returns"].rolling(20).apply(lambda x: x.autocorr(lag=lag), raw=False)
            )

        # Volume features (if available)
        if "Volume" in df.columns and df["Volume"].sum() > 0:
            features["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
            features["price_volume"] = features["returns"] * np.log(df["Volume"] + 1)
        else:
            features["volume_ratio"] = np.ones(len(df))
            features["price_volume"] = features["returns"]

        # Cross-correlations between key features
        corr_window = 20
        features["ret_vol_corr"] = (
            features["returns"].rolling(corr_window).corr(features["volatility"])
        )
        features["trend_vol_corr"] = (
            features["trend_strength"].rolling(corr_window).corr(features["volatility"])
        )

        # Statistical arbitrage features
        features["price_zscore"] = (df["Close"] - df["Close"].rolling(50).mean()) / (
            df["Close"].rolling(50).std() + 1e-8
        )
        features["return_zscore"] = (
            features["returns"] - features["returns"].rolling(50).mean()
        ) / (features["returns"].rolling(50).std() + 1e-8)

        # Drop rows with NaN values
        features = features.dropna()

        # Store feature names for later use
        self.feature_names = list(features.columns)

        return features

    def fit(self, df: pd.DataFrame) -> "HiddenMarkovRegimeDetector":
        """
        Train the HMM using Gaussian Mixture Models.

        This method implements the core training logic following
        Renaissance Technologies' approach with proper transition
        matrix estimation.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Self for method chaining

        Raises:
            ValueError: If fitting fails due to insufficient data
        """
        try:
            # Prepare features
            X = self._prepare_features(df)

            if len(X) < self.n_states * 5:
                raise ValueError(f"Insufficient data for {self.n_states} states")

            # Standardize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Fit Gaussian Mixture Model
            self.gmm = GaussianMixture(
                n_components=self.n_states,
                covariance_type="full",
                max_iter=200,
                n_init=5,
                random_state=42,
            )

            # Fit and predict states
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                states = self.gmm.fit_predict(X_scaled)

            # Estimate transition matrix
            self.transition_matrix = self._estimate_transition_matrix(states)

            # Store state characteristics for regime mapping. A component that
            # received no hard assignments falls back to its fitted GMM mean
            # instead of producing a NaN row.
            self.state_means = np.array(
                [
                    X_scaled[states == i].mean(axis=0)
                    if np.any(states == i)
                    else self.gmm.means_[i]
                    for i in range(self.n_states)
                ]
            )

            self.fitted = True

            return self

        except Exception as e:
            raise ValueError(f"HMM fitting failed: {e!s}") from e

    def _estimate_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        Calculate state transition probabilities with Laplace smoothing.

        One pseudo-count is added to every cell, so every row is a proper
        probability distribution: a state that is never left (or never visited)
        gets a uniform row instead of an all-zero one.

        Args:
            states: Array of state sequences

        Returns:
            Row-stochastic transition probability matrix
        """
        transition_counts = np.ones((self.n_states, self.n_states))

        for i in range(len(states) - 1):
            transition_counts[int(states[i]), int(states[i + 1])] += 1

        row_sums = transition_counts.sum(axis=1, keepdims=True)
        return np.asarray(transition_counts / row_sums)

    def _feature_index(self, name: str) -> int | None:
        """Return the column index of feature ``name`` or None if absent."""
        try:
            return self.feature_names.index(name)
        except ValueError:
            return None

    def _raw_feature_mean(self, recent: np.ndarray, name: str) -> float | None:
        """
        Mean of feature ``name`` over ``recent`` rows in original (unscaled) units.

        Returns None if the feature is unknown. Falls back to the z-value when
        the scaler is unavailable.
        """
        idx = self._feature_index(name)
        if idx is None or idx >= recent.shape[1]:
            return None
        z = float(recent[:, idx].mean())
        if self.scaler is None or not hasattr(self.scaler, "scale_"):
            return z
        return z * float(self.scaler.scale_[idx]) + float(self.scaler.mean_[idx])

    def _map_states_to_regimes(self, X: np.ndarray, states: np.ndarray) -> MarketRegime:
        """
        Map mathematical states to interpretable market regimes.

        The current state is characterised by the mean of the standardized
        features over the last ``_RECENT_WINDOW`` bars spent in that state.
        Features are looked up by name (``self.feature_names``), and all
        thresholds are in z-units because ``X`` is standardized:

        - volatility above / below the 75th / 25th percentile -> HIGH / LOW_VOLATILITY
        - trend_strength > +0.2 sd with above-average returns -> BULL_TRENDING
        - trend_strength < -0.2 sd with below-average returns -> BEAR_TRENDING
        - |trend_strength| < 0.2 sd and negative raw lag-1 autocorrelation
          -> MEAN_REVERTING
        - |returns| > 0.5 sd -> BREAKOUT
        - otherwise UNKNOWN

        Args:
            X: Standardized feature matrix (columns ordered as ``feature_names``)
            states: State predictions, one per row of ``X``

        Returns:
            MarketRegime classification
        """
        if len(states) == 0 or len(X) == 0:
            return MarketRegime.UNKNOWN

        ret_idx = self._feature_index("returns")
        vol_idx = self._feature_index("volatility")
        trend_idx = self._feature_index("trend_strength")
        if ret_idx is None or vol_idx is None or trend_idx is None:
            return MarketRegime.UNKNOWN

        current_state = states[-1]
        current_state_mask = states[-_RECENT_WINDOW:] == current_state
        recent_data = X[-_RECENT_WINDOW:][current_state_mask]
        if len(recent_data) == 0:
            return MarketRegime.UNKNOWN

        avg_returns = float(recent_data[:, ret_idx].mean())
        avg_volatility = float(recent_data[:, vol_idx].mean())
        avg_trend = float(recent_data[:, trend_idx].mean())
        raw_autocorr = self._raw_feature_mean(recent_data, "autocorr_1")

        vol_threshold_high = float(np.percentile(X[:, vol_idx], 75))
        vol_threshold_low = float(np.percentile(X[:, vol_idx], 25))

        if avg_volatility > vol_threshold_high:
            return MarketRegime.HIGH_VOLATILITY
        if avg_volatility < vol_threshold_low:
            return MarketRegime.LOW_VOLATILITY
        if avg_trend > _TREND_Z and avg_returns > 0:
            return MarketRegime.BULL_TRENDING
        if avg_trend < -_TREND_Z and avg_returns < 0:
            return MarketRegime.BEAR_TRENDING
        if abs(avg_trend) < _TREND_Z and raw_autocorr is not None and raw_autocorr < 0:
            return MarketRegime.MEAN_REVERTING
        if abs(avg_returns) > _BREAKOUT_Z:
            return MarketRegime.BREAKOUT
        return MarketRegime.UNKNOWN

    def predict_with_states(self, df: pd.DataFrame) -> tuple[MarketRegime, np.ndarray, float]:
        """
        Predict the current regime and the full state sequence in one pass.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (regime, states, confidence) where ``states`` holds one
            int state per feature row (the last element is the current state)
            and ``confidence`` is the GMM posterior of the current state.

        Raises:
            ValueError: If model is not fitted or prediction fails
        """
        if not self.fitted or self.scaler is None or self.gmm is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            X = self._prepare_features(df)
            X_scaled = self.scaler.transform(X)

            states = self.gmm.predict(X_scaled).astype(int)
            probabilities = self.gmm.predict_proba(X_scaled)

            regime = self._map_states_to_regimes(X_scaled, states)
            confidence = float(probabilities[-1].max())

            return regime, states, confidence

        except Exception as e:
            raise ValueError(f"Regime prediction failed: {e!s}") from e

    def predict_regime(self, df: pd.DataFrame) -> tuple[MarketRegime, int, float]:
        """
        Predict the current market regime.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (regime, state, confidence) as plain Python types

        Raises:
            ValueError: If model is not fitted or prediction fails
        """
        regime, states, confidence = self.predict_with_states(df)
        return regime, int(states[-1]), confidence

    def get_transition_probability(self, current_state: int, target_state: int) -> float:
        """
        Get the probability of transitioning from current to target state.

        Edge cases match :meth:`TrueHMMDetector.get_transition_probability`.

        Args:
            current_state: Current HMM state
            target_state: Target HMM state

        Returns:
            Transition probability (0-1)

        Raises:
            ValueError: If the model is not fitted or a state index is out of range
        """
        if not self.fitted or self.transition_matrix is None:
            raise ValueError("Model must be fitted first")

        if current_state < 0 or current_state >= self.n_states:
            raise ValueError(f"Invalid from_state: {current_state}")

        if target_state < 0 or target_state >= self.n_states:
            raise ValueError(f"Invalid to_state: {target_state}")

        return float(self.transition_matrix[current_state, target_state])

    def calculate_regime_persistence(self, states: np.ndarray, lookback: int = 20) -> float:
        """
        Calculate regime stability metric.

        Fraction of the last ``lookback`` states equal to the current (last)
        state. Fewer than two observations carry no information about
        persistence and return 0.0. Same behavior as
        :meth:`TrueHMMDetector.calculate_regime_persistence`.

        Args:
            states: Array of recent state predictions
            lookback: Number of periods to look back

        Returns:
            Persistence score (0-1, higher = more stable)
        """
        lookback = min(lookback, len(states))

        if lookback < 2:
            return 0.0

        recent_states = np.asarray(states)[-lookback:]
        return float(np.mean(recent_states == recent_states[-1]))
