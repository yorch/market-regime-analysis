"""
Hidden Markov Model regime detector implementation.

This module implements the core HMM functionality following Jim Simons'
mathematical approach for market regime detection.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .enums import MarketRegime

warnings.filterwarnings("ignore", category=UserWarning)


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
            states = self.gmm.fit_predict(X_scaled)

            # Estimate transition matrix
            self.transition_matrix = self._estimate_transition_matrix(states)

            # Store state characteristics for regime mapping
            self.state_means = np.array(
                [X_scaled[states == i].mean(axis=0) for i in range(self.n_states)]
            )

            self.fitted = True

            return self

        except Exception as e:
            raise ValueError(f"HMM fitting failed: {e!s}")

    def _estimate_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        Calculate state transition probabilities.

        Args:
            states: Array of state sequences

        Returns:
            Transition probability matrix
        """
        transition_counts = np.zeros((self.n_states, self.n_states))

        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_counts[current_state, next_state] += 1

        # Normalize to get probabilities (add small epsilon for stability)
        row_sums = transition_counts.sum(axis=1) + 1e-8
        transition_matrix = transition_counts / row_sums[:, np.newaxis]

        return transition_matrix

    def _map_states_to_regimes(self, X: np.ndarray, states: np.ndarray) -> MarketRegime:
        """
        Map mathematical states to interpretable market regimes.

        This method uses the statistical characteristics of each state
        to assign meaningful regime labels.

        Args:
            X: Feature matrix
            states: State predictions

        Returns:
            MarketRegime classification
        """
        if len(states) == 0:
            return MarketRegime.UNKNOWN

        # Get the most recent state
        current_state = states[-1]

        # Get recent data for the current state
        current_state_mask = states[-20:] == current_state
        if not any(current_state_mask):
            return MarketRegime.UNKNOWN

        # Analyze characteristics of current state
        recent_data = X[-20:][current_state_mask]

        if len(recent_data) == 0:
            return MarketRegime.UNKNOWN

        # Extract key features for regime classification
        avg_returns = recent_data[:, 0].mean()  # returns
        avg_volatility = recent_data[:, 4].mean()  # volatility
        avg_trend = recent_data[:, 9].mean() if recent_data.shape[1] > 9 else 0  # trend_strength

        # Regime classification logic
        vol_threshold_high = np.percentile(X[:, 4], 75)
        vol_threshold_low = np.percentile(X[:, 4], 25)

        if avg_volatility > vol_threshold_high:
            return MarketRegime.HIGH_VOLATILITY
        elif avg_volatility < vol_threshold_low:
            return MarketRegime.LOW_VOLATILITY
        elif avg_returns > 0.001 and avg_trend > 0:
            return MarketRegime.BULL_TRENDING
        elif avg_returns < -0.001 and avg_trend < 0:
            return MarketRegime.BEAR_TRENDING
        elif abs(avg_trend) < 0.001:
            return MarketRegime.MEAN_REVERTING
        elif abs(avg_returns) > 0.002:
            return MarketRegime.BREAKOUT
        else:
            return MarketRegime.UNKNOWN

    def predict_regime(self, df: pd.DataFrame) -> tuple[MarketRegime, int, float]:
        """
        Predict the current market regime.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (regime, state, confidence)

        Raises:
            ValueError: If model is not fitted or prediction fails
        """
        if not self.fitted or self.scaler is None or self.gmm is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Prepare features
            X = self._prepare_features(df)
            X_scaled = self.scaler.transform(X)

            # Predict states
            states = self.gmm.predict(X_scaled)
            probabilities = self.gmm.predict_proba(X_scaled)

            # Map to regime
            regime = self._map_states_to_regimes(X_scaled, states)

            # Calculate confidence as max probability
            confidence = probabilities[-1].max()

            return regime, states[-1], confidence

        except Exception as e:
            raise ValueError(f"Regime prediction failed: {e!s}")

    def get_transition_probability(self, current_state: int, target_state: int) -> float:
        """
        Get the probability of transitioning from current to target state.

        Args:
            current_state: Current HMM state
            target_state: Target HMM state

        Returns:
            Transition probability
        """
        if not self.fitted or self.transition_matrix is None:
            return 0.0

        if (
            current_state < 0
            or current_state >= self.n_states
            or target_state < 0
            or target_state >= self.n_states
        ):
            return 0.0

        return float(self.transition_matrix[current_state, target_state])

    def calculate_regime_persistence(self, states: np.ndarray, lookback: int = 20) -> float:
        """
        Calculate regime stability metric.

        This measures how stable the current regime is by looking
        at the consistency of recent state predictions.

        Args:
            states: Array of recent state predictions
            lookback: Number of periods to look back

        Returns:
            Persistence score (0-1, higher = more stable)
        """
        lookback = min(lookback, len(states))

        if lookback <= 1:
            return 0.0

        recent_states = states[-lookback:]
        current_state = recent_states[-1]

        # Calculate percentage of recent periods in current state
        persistence = np.sum(recent_states == current_state) / len(recent_states)

        return float(persistence)
