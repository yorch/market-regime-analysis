"""
True Hidden Markov Model implementation using hmmlearn.

This module implements a proper HMM with temporal dependencies,
using the Baum-Welch algorithm for training and Viterbi for decoding.
This addresses the critical flaw in the original GMM-based approach.
"""

import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from .enums import MarketRegime

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TrueHMMDetector:
    """
    Proper HMM implementation using hmmlearn library.

    This class implements a true Hidden Markov Model with:
    - Gaussian emission distributions
    - Baum-Welch algorithm for parameter learning
    - Viterbi algorithm for optimal state sequence decoding
    - Proper temporal dependency modeling (unlike GMM)

    Key Differences from GMM Approach:
    - Models state transitions over time (temporal dependencies)
    - Uses forward-backward algorithm for probability estimation
    - Learns transition matrix as part of training (not post-hoc)
    - Produces coherent state sequences respecting dynamics
    """

    def __init__(
        self,
        n_states: int = 6,
        n_iter: int = 100,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        """
        Initialize the True HMM detector.

        Args:
            n_states: Number of hidden states (default 6 for regime detection)
            n_iter: Maximum iterations for Baum-Welch training
            covariance_type: Type of covariance ('full', 'diag', 'tied', 'spherical')
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        # HMM model
        self.model: hmm.GaussianHMM | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self.fitted: bool = False

        # Learned parameters
        self.transition_matrix: np.ndarray | None = None
        self.state_means: np.ndarray | None = None
        self.state_covariances: np.ndarray | None = None
        self.training_score: float | None = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive features for HMM analysis.

        Uses same features as original detector for fair comparison,
        but calculates them without look-ahead bias.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features (no NaN values)

        Raises:
            ValueError: If insufficient data
        """
        if len(df) < 50:
            raise ValueError("Insufficient data for feature calculation (minimum 50 bars)")

        features = pd.DataFrame(index=df.index)

        # Basic price features (shifted to avoid look-ahead)
        features["returns"] = df["Close"].pct_change()
        features["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # Volatility features (using expanding window to avoid bias)
        features["volatility"] = features["returns"].rolling(20, min_periods=10).std()
        features["log_volatility"] = np.log(features["volatility"] + 1e-8)

        # ATR calculation
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift(1))
        low_close = np.abs(df["Low"] - df["Close"].shift(1))
        true_range = pd.Series(
            np.maximum(high_low, np.maximum(high_close, low_close)), index=df.index
        )
        features["atr"] = true_range.rolling(14, min_periods=7).mean()
        features["atr_normalized"] = features["atr"] / df["Close"]

        # Higher-order moments (with minimum sample size)
        window = 20
        features["skewness"] = features["returns"].rolling(window, min_periods=window).skew()
        features["kurtosis"] = features["returns"].rolling(window, min_periods=window).kurt()

        # Trend strength features
        features["sma_9"] = df["Close"].rolling(9, min_periods=5).mean()
        features["sma_21"] = df["Close"].rolling(21, min_periods=10).mean()
        features["sma_50"] = df["Close"].rolling(50, min_periods=25).mean()

        features["trend_9_21"] = (features["sma_9"] - features["sma_21"]) / df["Close"]
        features["trend_21_50"] = (features["sma_21"] - features["sma_50"]) / df["Close"]

        # Autocorrelation with proper sample size
        for lag in [1, 2, 5]:
            features[f"autocorr_{lag}"] = features["returns"].rolling(60, min_periods=40).apply(
                lambda x: x.autocorr(lag=lag) if len(x.dropna()) >= lag + 20 else np.nan,
                raw=False,
            )

        # Volume features (if available)
        if "Volume" in df.columns and df["Volume"].sum() > 0:
            features["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20, min_periods=10).mean()
        else:
            features["volume_ratio"] = 1.0

        # Price Z-score (using rolling mean/std - shifted to avoid bias)
        rolling_mean = df["Close"].rolling(50, min_periods=25).mean()
        rolling_std = df["Close"].rolling(50, min_periods=25).std()
        features["price_zscore"] = (df["Close"] - rolling_mean) / (rolling_std + 1e-8)

        # Cross-feature relationships
        features["return_vol_ratio"] = features["returns"] / (features["volatility"] + 1e-8)
        features["trend_vol_interaction"] = features["trend_9_21"] * features["volatility"]

        # Drop NaN values and save feature names
        features = features.dropna()
        self.feature_names = list(features.columns)

        return features

    def fit(self, df: pd.DataFrame) -> "TrueHMMDetector":
        """
        Train HMM using Baum-Welch algorithm.

        This is a proper HMM training that:
        1. Initializes transition and emission parameters
        2. Uses Baum-Welch (EM algorithm) to optimize parameters
        3. Learns temporal dependencies in state transitions

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Self for method chaining

        Raises:
            ValueError: If fitting fails
        """
        try:
            # Prepare features
            X = self._prepare_features(df)

            if len(X) < self.n_states * 10:
                raise ValueError(
                    f"Insufficient data for {self.n_states} states "
                    f"(need at least {self.n_states * 10}, got {len(X)})"
                )

            # Standardize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Initialize and train Gaussian HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
                verbose=False,
            )

            # Fit using Baum-Welch algorithm
            self.model.fit(X_scaled)

            # Store learned parameters
            self.transition_matrix = self.model.transmat_
            self.state_means = self.model.means_
            self.state_covariances = self.model.covars_

            # Calculate training log-likelihood (goodness of fit)
            self.training_score = self.model.score(X_scaled)

            self.fitted = True

            return self

        except Exception as e:
            raise ValueError(f"HMM fitting failed: {e!s}") from e

    def predict_regime(
        self, df: pd.DataFrame, use_viterbi: bool = True
    ) -> tuple[MarketRegime, int, float]:
        """
        Predict market regime using trained HMM.

        Args:
            df: DataFrame with OHLCV data
            use_viterbi: If True, use Viterbi algorithm for optimal state sequence.
                         If False, use forward algorithm for marginal probabilities.

        Returns:
            Tuple of (regime, state, confidence)

        Raises:
            ValueError: If model not fitted
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Prepare features
            X = self._prepare_features(df)
            X_scaled = self.scaler.transform(X)

            if use_viterbi:
                # Viterbi: optimal state sequence considering full history
                states = self.model.predict(X_scaled)
            else:
                # Forward algorithm: marginal probabilities for each time step
                state_probs = self.model.predict_proba(X_scaled)
                states = np.argmax(state_probs, axis=1)

            # Get current state
            current_state = int(states[-1])

            # Calculate confidence using state probability from forward algorithm
            # This gives us P(state_t | observations_1:t)
            if len(X_scaled) > 0:
                # Get probability distribution over states for most recent observation
                state_probs = self.model.predict_proba(X_scaled)
                # Confidence is probability of being in the predicted state
                confidence = float(state_probs[-1][current_state])
            else:
                confidence = 0.0

            # Map state to interpretable regime
            regime = self._map_state_to_regime(X_scaled, states, current_state)

            return regime, current_state, confidence

        except Exception as e:
            raise ValueError(f"Prediction failed: {e!s}") from e

    def _map_state_to_regime(
        self, X: np.ndarray, states: np.ndarray, current_state: int
    ) -> MarketRegime:
        """
        Map HMM state to interpretable market regime.

        Uses state characteristics to assign regime labels based on:
        - Returns (positive/negative)
        - Volatility (high/low)
        - Trend strength
        - Autocorrelation (momentum vs mean-reversion)

        Args:
            X: Scaled feature matrix
            states: Predicted state sequence
            current_state: Current state index

        Returns:
            MarketRegime classification
        """
        if self.state_means is None:
            return MarketRegime.UNKNOWN

        try:
            # Get state characteristics from learned emission means
            state_features = self.state_means[current_state]

            # Map features back to original indices
            # Assume feature order: returns, log_returns, volatility, ...
            # We need to be careful with indices
            feature_dict = {}
            for idx, name in enumerate(self.feature_names):
                if idx < len(state_features):
                    feature_dict[name] = state_features[idx]

            # Extract key features (already standardized - values around 0 mean, 1 std)
            avg_returns = feature_dict.get("returns", 0.0)
            avg_volatility = feature_dict.get("volatility", 0.0)
            avg_trend = feature_dict.get("trend_9_21", 0.0)
            avg_autocorr = feature_dict.get("autocorr_1", 0.0)

            # Calculate volatility percentiles across all states
            # Find volatility feature index
            vol_idx = None
            for idx, name in enumerate(self.feature_names):
                if name == "volatility":
                    vol_idx = idx
                    break

            if vol_idx is not None:
                all_volatilities = [self.state_means[i, vol_idx] for i in range(self.n_states)]
                vol_threshold_high = np.percentile(all_volatilities, 75)
                vol_threshold_low = np.percentile(all_volatilities, 25)
            else:
                vol_threshold_high = 0.5
                vol_threshold_low = -0.5

            # Regime classification logic (adjusted for standardized values)
            # Since features are standardized, use more appropriate thresholds
            if avg_volatility > vol_threshold_high:
                return MarketRegime.HIGH_VOLATILITY
            elif avg_volatility < vol_threshold_low:
                return MarketRegime.LOW_VOLATILITY
            elif avg_returns > 0.2 and avg_trend > 0.2:  # Above mean trending up
                return MarketRegime.BULL_TRENDING
            elif avg_returns < -0.2 and avg_trend < -0.2:  # Below mean trending down
                return MarketRegime.BEAR_TRENDING
            elif abs(avg_autocorr) < 0.3:  # Low autocorrelation suggests mean reversion
                return MarketRegime.MEAN_REVERTING
            elif avg_volatility > 0.4:  # Above average volatility
                return MarketRegime.BREAKOUT
            else:
                return MarketRegime.UNKNOWN

        except Exception:
            return MarketRegime.UNKNOWN

    def calculate_regime_persistence(self, states: np.ndarray, lookback: int = 20) -> float:
        """
        Calculate regime stability metric.

        Measures what percentage of recent observations remained in same state.

        Args:
            states: State sequence
            lookback: Number of recent periods to examine

        Returns:
            Persistence score (0-1, higher = more stable)
        """
        if len(states) < lookback:
            lookback = len(states)

        if lookback == 0:
            return 0.0

        recent_states = states[-lookback:]
        current_state = states[-1]

        persistence = np.mean(recent_states == current_state)
        return float(persistence)

    def get_transition_probability(self, from_state: int, to_state: int) -> float:
        """
        Get learned transition probability between states.

        Args:
            from_state: Source state index
            to_state: Target state index

        Returns:
            Transition probability (0-1)

        Raises:
            ValueError: If model not fitted
        """
        if not self.fitted or self.transition_matrix is None:
            raise ValueError("Model must be fitted first")

        if from_state < 0 or from_state >= self.n_states:
            raise ValueError(f"Invalid from_state: {from_state}")

        if to_state < 0 or to_state >= self.n_states:
            raise ValueError(f"Invalid to_state: {to_state}")

        return float(self.transition_matrix[from_state, to_state])

    def get_training_convergence(self) -> dict:
        """
        Get HMM training convergence information.

        Returns:
            Dictionary with training metrics including log-likelihood
        """
        if not self.fitted or self.model is None:
            return {"fitted": False}

        return {
            "fitted": True,
            "log_likelihood": self.training_score,
            "n_states": self.n_states,
            "n_features": len(self.feature_names),
            "covariance_type": self.covariance_type,
            "converged": self.model.monitor_.converged,
            "n_iterations": len(self.model.monitor_.history) if hasattr(self.model.monitor_, "history") else "N/A",
        }

    def compare_with_gmm(self, df: pd.DataFrame, gmm_detector) -> dict:
        """
        Compare predictions with GMM-based detector.

        Args:
            df: DataFrame with OHLCV data
            gmm_detector: Instance of HiddenMarkovRegimeDetector (GMM-based)

        Returns:
            Dictionary with comparison metrics
        """
        # Get HMM predictions
        hmm_regime, hmm_state, hmm_confidence = self.predict_regime(df)

        # Get GMM predictions
        gmm_regime, gmm_state, gmm_confidence = gmm_detector.predict_regime(df)

        return {
            "hmm_regime": hmm_regime.value,
            "hmm_state": hmm_state,
            "hmm_confidence": hmm_confidence,
            "gmm_regime": gmm_regime.value,
            "gmm_state": gmm_state,
            "gmm_confidence": gmm_confidence,
            "regime_agreement": hmm_regime == gmm_regime,
            "state_agreement": hmm_state == gmm_state,
        }
