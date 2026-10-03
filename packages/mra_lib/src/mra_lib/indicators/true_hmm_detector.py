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

from mra_lib.config.enums import MarketRegime

# Regime thresholds for standardized (z-unit) state means.
_TREND_Z = 0.2
_BREAKOUT_VOL_Z = 0.4


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

        Every feature is causal: it is computed from trailing rolling windows
        that only use the current and earlier bars (no look-ahead).

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

        # Basic price features
        features["returns"] = df["Close"].pct_change()
        features["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # Volatility features (trailing 20-bar window, at least 10 observations)
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

        # Autocorrelation — vectorized for performance
        returns = features["returns"]
        for lag in [1, 5]:
            lagged = returns.shift(lag)
            roll_cov = returns.rolling(30, min_periods=20).cov(lagged)
            roll_var = returns.rolling(30, min_periods=20).var()
            features[f"autocorr_{lag}"] = roll_cov / (roll_var + 1e-12)

        # Volume features (if available)
        if "Volume" in df.columns and df["Volume"].sum() > 0:
            features["volume_ratio"] = (
                df["Volume"] / df["Volume"].rolling(20, min_periods=10).mean()
            )
        else:
            features["volume_ratio"] = 1.0

        # Price Z-score against a trailing 50-bar mean/std (current bar included)
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
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

    def _require_fitted(self) -> tuple[hmm.GaussianHMM, StandardScaler, np.ndarray]:
        """
        Return the fitted model, scaler and transition matrix.

        Raises:
            ValueError: If the detector has not been fitted
        """
        if (
            not self.fitted
            or self.model is None
            or self.scaler is None
            or self.transition_matrix is None
        ):
            raise ValueError("Model must be fitted before use")
        return self.model, self.scaler, self.transition_matrix

    def _current_posterior(self, df: pd.DataFrame) -> np.ndarray:
        """
        Posterior state distribution for the most recent bar.

        ``GaussianHMM.predict_proba`` returns forward-backward (smoothed)
        posteriors P(s_t | o_1..o_T). For the last bar t = T, so the smoothed
        and filtered posteriors coincide and this is P(s_T | o_1..o_T).
        """
        model, scaler, _ = self._require_fitted()
        X_scaled = scaler.transform(self._prepare_features(df))
        posterior: np.ndarray = model.predict_proba(X_scaled)[-1]
        return posterior

    def predict_regime(
        self, df: pd.DataFrame, use_viterbi: bool = True
    ) -> tuple[MarketRegime, int, float]:
        """
        Predict market regime using trained HMM.

        Args:
            df: DataFrame with OHLCV data
            use_viterbi: If True, decode the most likely state sequence with
                Viterbi. If False, take the argmax of the per-bar smoothed
                (forward-backward) posteriors.

        Returns:
            Tuple of (regime, state, confidence). ``confidence`` is the
            posterior probability of the returned state at the last bar
            (smoothed == filtered at the final time step).

        Raises:
            ValueError: If model not fitted or prediction fails
        """
        model, scaler, _ = self._require_fitted()

        try:
            X_scaled = scaler.transform(self._prepare_features(df))

            # Smoothed posteriors P(s_t | o_1..o_T) for every bar.
            state_probs = model.predict_proba(X_scaled)

            # Viterbi path, or per-bar argmax of the smoothed posteriors
            states = model.predict(X_scaled) if use_viterbi else np.argmax(state_probs, axis=1)

            current_state = int(states[-1])
            confidence = float(state_probs[-1][current_state])
            regime = self._map_state_index_to_regime(current_state)

            return regime, current_state, confidence

        except Exception as e:
            raise ValueError(f"Prediction failed: {e!s}") from e

    def _map_state_to_regime(
        self, X: np.ndarray, states: np.ndarray, current_state: int
    ) -> MarketRegime:
        """
        Map the current HMM state to a regime.

        Kept for backward compatibility; the label depends only on the learned
        emission means, so this delegates to :meth:`_map_state_index_to_regime`
        (``X`` and ``states`` are unused).
        """
        return self._map_state_index_to_regime(current_state)

    def calculate_regime_persistence(self, states: np.ndarray, lookback: int = 20) -> float:
        """
        Calculate regime stability metric.

        Fraction of the last ``lookback`` states equal to the current (last)
        state. Fewer than two observations carry no information about
        persistence and return 0.0 (same as the GMM detector).

        Args:
            states: State sequence
            lookback: Number of recent periods to examine

        Returns:
            Persistence score (0-1, higher = more stable)
        """
        lookback = min(lookback, len(states))

        if lookback < 2:
            return 0.0

        recent_states = np.asarray(states)[-lookback:]
        return float(np.mean(recent_states == recent_states[-1]))

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
            "n_iterations": len(self.model.monitor_.history)
            if hasattr(self.model.monitor_, "history")
            else "N/A",
        }

    def _map_state_index_to_regime(self, state_index: int) -> MarketRegime:
        """
        Map a state index to a MarketRegime using learned emission means only.

        Does not require observed data, so it is also used for forecasting.
        Features are looked up by name; thresholds are in z-units because the
        emission means live in standardized feature space:

        - volatility above / below the 75th / 25th percentile of state means
          -> HIGH / LOW_VOLATILITY
        - returns and trend_9_21 both > +0.2 sd -> BULL_TRENDING
        - returns and trend_9_21 both < -0.2 sd -> BEAR_TRENDING
        - negative lag-1 autocorrelation in original units -> MEAN_REVERTING
        - volatility > +0.4 sd -> BREAKOUT
        - otherwise UNKNOWN

        Args:
            state_index: HMM state index

        Returns:
            MarketRegime classification
        """
        if self.state_means is None or not 0 <= state_index < self.n_states:
            return MarketRegime.UNKNOWN

        try:
            state_features = self.state_means[state_index]
            feature_dict = {
                name: float(state_features[idx])
                for idx, name in enumerate(self.feature_names)
                if idx < len(state_features)
            }

            avg_returns = feature_dict.get("returns", 0.0)
            avg_volatility = feature_dict.get("volatility", 0.0)
            avg_trend = feature_dict.get("trend_9_21", 0.0)

            # Lag-1 autocorrelation back in original units so its sign is meaningful.
            raw_autocorr: float | None = None
            if "autocorr_1" in self.feature_names and self.scaler is not None:
                ac_idx = self.feature_names.index("autocorr_1")
                raw_autocorr = feature_dict.get("autocorr_1", 0.0) * float(
                    self.scaler.scale_[ac_idx]
                ) + float(self.scaler.mean_[ac_idx])

            if "volatility" in self.feature_names:
                vol_idx = self.feature_names.index("volatility")
                all_vols = self.state_means[:, vol_idx]
                vol_high = float(np.percentile(all_vols, 75))
                vol_low = float(np.percentile(all_vols, 25))
            else:
                vol_high, vol_low = 0.5, -0.5

            if avg_volatility > vol_high:
                return MarketRegime.HIGH_VOLATILITY
            if avg_volatility < vol_low:
                return MarketRegime.LOW_VOLATILITY
            if avg_returns > _TREND_Z and avg_trend > _TREND_Z:
                return MarketRegime.BULL_TRENDING
            if avg_returns < -_TREND_Z and avg_trend < -_TREND_Z:
                return MarketRegime.BEAR_TRENDING
            if raw_autocorr is not None and raw_autocorr < 0:
                return MarketRegime.MEAN_REVERTING
            if avg_volatility > _BREAKOUT_VOL_Z:
                return MarketRegime.BREAKOUT
            return MarketRegime.UNKNOWN

        except Exception:
            return MarketRegime.UNKNOWN

    def get_state_regime_map(self) -> dict[int, MarketRegime]:
        """
        Get the mapping from all state indices to MarketRegime.

        Returns:
            Dictionary mapping state index to MarketRegime

        Raises:
            ValueError: If model not fitted
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting state map")

        return {i: self._map_state_index_to_regime(i) for i in range(self.n_states)}

    def forecast_regime_probabilities(self, df: pd.DataFrame, n_steps: int = 1) -> np.ndarray:
        """
        Forecast regime state probability distribution n steps ahead.

        Uses the current posterior state distribution and the learned
        transition matrix to project future state probabilities:
            pi_{t+n} = pi_t @ T^n

        Args:
            df: DataFrame with OHLCV data (used to determine current state)
            n_steps: Number of steps ahead to forecast (default 1)

        Returns:
            Array of shape (n_states,) with forecasted state probabilities

        Raises:
            ValueError: If model not fitted or n_steps < 1
        """
        _, _, transmat = self._require_fitted()
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")

        pi_t = self._current_posterior(df)

        # Project forward: pi_{t+n} = pi_t @ T^n
        forecast: np.ndarray = pi_t @ np.linalg.matrix_power(transmat, n_steps)
        return forecast

    def forecast_regime_sequence(self, df: pd.DataFrame, n_steps: int = 5) -> list[dict]:
        """
        Forecast regime probabilities for each step from 1 to n_steps.

        For each forecast horizon, produces the probability distribution
        over regimes (aggregated from HMM states) and the most likely regime.

        Args:
            df: DataFrame with OHLCV data
            n_steps: Number of steps to forecast (default 5)

        Returns:
            List of dicts, one per step, each containing:
                - step: forecast horizon (1-indexed)
                - state_probabilities: raw state probability array
                - regime_probabilities: dict[MarketRegime, float]
                - most_likely_regime: MarketRegime with highest probability
                - most_likely_regime_probability: float
        """
        _, _, transmat = self._require_fitted()
        pi_t = self._current_posterior(df)

        # Build state-to-regime mapping once
        state_regime_map = self.get_state_regime_map()

        results = []
        for step in range(1, n_steps + 1):
            forecast_probs = pi_t @ np.linalg.matrix_power(transmat, step)

            # Aggregate state probs by regime
            regime_probs: dict[MarketRegime, float] = {}
            for state_idx, prob in enumerate(forecast_probs):
                regime = state_regime_map[state_idx]
                regime_probs[regime] = regime_probs.get(regime, 0.0) + float(prob)

            most_likely = max(regime_probs, key=lambda r: regime_probs[r])

            results.append(
                {
                    "step": step,
                    "state_probabilities": forecast_probs,
                    "regime_probabilities": regime_probs,
                    "most_likely_regime": most_likely,
                    "most_likely_regime_probability": regime_probs[most_likely],
                }
            )

        return results

    def get_regime_stability(self) -> dict:
        """
        Compute regime stability metrics from the learned transition matrix.

        Returns a dictionary with:
        - self_transition_probs: diagonal of T (probability of staying in each state)
        - expected_durations: expected number of steps in each state (1 / (1 - T_ii))
        - stationary_distribution: long-run state probabilities (left eigenvector of T)
        - stationary_regimes: stationary distribution aggregated by MarketRegime

        Raises:
            ValueError: If model not fitted
        """
        if not self.fitted or self.transition_matrix is None:
            raise ValueError("Model must be fitted before computing stability")

        T = self.transition_matrix

        # Self-transition probabilities (diagonal)
        self_trans = {i: float(T[i, i]) for i in range(self.n_states)}

        # Expected duration in each state: 1 / (1 - p_ii)
        expected_dur = {}
        for i in range(self.n_states):
            p_ii = T[i, i]
            expected_dur[i] = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else float("inf")

        # Stationary distribution: left eigenvector of T (pi @ T = pi)
        # Equivalent to right eigenvector of T^T with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(T.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()  # Normalize to probability

        # Aggregate by regime
        state_regime_map = self.get_state_regime_map()
        stationary_regimes: dict[MarketRegime, float] = {}
        for state_idx, prob in enumerate(stationary):
            regime = state_regime_map[state_idx]
            stationary_regimes[regime] = stationary_regimes.get(regime, 0.0) + prob

        return {
            "self_transition_probs": self_trans,
            "expected_durations": expected_dur,
            "stationary_distribution": stationary,
            "stationary_regimes": stationary_regimes,
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
