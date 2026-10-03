"""
Gaussian hidden Markov model regime detector (hmmlearn).

This is the single regime model used across the library: the analyzer, the
CLI forecast command, walk-forward validation and the optimizer all fit and
query :class:`TrueHMMDetector`, so validation numbers describe the model users
actually see.

Design notes
------------
- **Features** come from :func:`mra_lib.indicators.features.build_hmm_features`:
  six causal, stationary, scale-free features (no raw price levels).
- **Model size**: diagonal covariances by default, ``min_covar`` regularization,
  several EM restarts (``n_init``) keeping the best log-likelihood, and a
  minimum-sample check scaled to the number of free parameters.
- **Canonical state order**: after fitting, states are sorted by ascending
  volatility, so state indices (and their labels) are stable across refits
  and seeds when the fitted solution is the same.
- **Regime labels** come from absolute thresholds on de-standardized state
  means (see :mod:`mra_lib.indicators.regime_mapping`).
- **Posteriors**: every per-bar quantity (state history, confidence) uses
  the *filtered* (forward-only) posterior ``P(s_t | o_1..o_t)``, which is
  causal in the observations. The model parameters (scaler, emissions,
  transitions, labels) are still in-sample for the data they were fitted on;
  only a model fitted on a prefix (as walk-forward does) is fully out-of-sample.
- **Short history**: if the data cannot support ``n_states`` and
  ``adapt_n_states`` is True (default), the largest state count the data can
  support is used instead (with a warning) rather than failing.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

from mra_lib.config.enums import MarketRegime

from .base import regime_persistence, transition_probability
from .features import HMM_FEATURES, HMM_WARMUP_BARS, build_hmm_features
from .regime_mapping import DEFAULT_THRESHOLDS, RegimeThresholds, StateSummary, classify_state

logger = logging.getLogger(__name__)

#: Minimum feature rows required per free model parameter.
MIN_SAMPLES_PER_PARAMETER = 1.5

_COVARIANCE_TYPES = ("diag", "full", "spherical", "tied")


def n_free_parameters(n_states: int, n_features: int, covariance_type: str) -> int:
    """
    Number of free parameters of a Gaussian HMM.

    ``(n-1)`` start probabilities + ``n(n-1)`` transition probabilities +
    ``n*f`` means + the covariance parameters for ``covariance_type``.
    """
    cov = {
        "diag": n_states * n_features,
        "full": n_states * n_features * (n_features + 1) // 2,
        "spherical": n_states,
        "tied": n_features * (n_features + 1) // 2,
    }
    if covariance_type not in cov:
        raise ValueError(
            f"Unknown covariance_type {covariance_type!r}; expected one of {_COVARIANCE_TYPES}"
        )
    return (n_states - 1) + n_states * (n_states - 1) + n_states * n_features + cov[covariance_type]


class _HmmlearnProgressFilter(logging.Filter):
    """Swallow hmmlearn's per-iteration 'not converging' warnings (summarized by us)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("Model is not converging")


@contextmanager
def _quiet_hmmlearn() -> Iterator[None]:
    """Scope-limited silencing of hmmlearn's noisy EM logging and deprecation warnings."""
    hmm_logger = logging.getLogger("hmmlearn.base")
    flt = _HmmlearnProgressFilter()
    hmm_logger.addFilter(flt)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            yield
    finally:
        hmm_logger.removeFilter(flt)


def forward_filter(model: hmm.GaussianHMM, X_scaled: np.ndarray) -> np.ndarray:
    """
    Filtered state posteriors ``P(s_t | o_1..o_t)`` for every row of ``X_scaled``.

    A log-space forward pass using hmmlearn's own emission densities, so the
    last row equals ``model.predict_proba(X_scaled)[-1]`` but no row uses
    observations after it (unlike ``predict_proba``, which is smoothed).

    Returns:
        Array of shape ``(len(X_scaled), n_states)``; rows sum to 1
    """
    log_lik = np.asarray(model._compute_log_likelihood(X_scaled))
    with np.errstate(divide="ignore"):
        log_start = np.log(np.asarray(model.startprob_))
        log_trans = np.log(np.asarray(model.transmat_))

    log_alpha = np.empty_like(log_lik)
    log_alpha[0] = log_start + log_lik[0]
    for t in range(1, len(log_lik)):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_lik[t]
    filtered: np.ndarray = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
    return filtered


class TrueHMMDetector:
    """
    Gaussian HMM market regime detector (Baum-Welch training, forward filtering).

    Args:
        n_states: Number of hidden states (see :func:`select_n_states` for a
            BIC-based choice)
        n_iter: Maximum EM iterations per restart
        covariance_type: ``'diag'`` (default), ``'full'``, ``'spherical'`` or ``'tied'``
        random_state: Seed of the first restart; restart ``k`` uses ``random_state + k``
        n_init: Number of EM restarts; the best log-likelihood wins
        min_covar: Variance floor added to the covariance diagonal (standardized units)
        tol: EM convergence threshold on the log-likelihood gain
        thresholds: State -> regime decision-tree thresholds
        min_samples_per_param: Feature rows required per free parameter
        adapt_n_states: If the data is too short for ``n_states``, fit the
            largest supported state count (>= 2) instead of raising
    """

    def __init__(  # noqa: PLR0913
        self,
        n_states: int = 6,
        n_iter: int = 200,
        covariance_type: str = "diag",
        random_state: int | None = 42,
        *,
        n_init: int = 10,
        min_covar: float = 1e-3,
        tol: float = 1e-2,
        thresholds: RegimeThresholds | None = None,
        min_samples_per_param: float = MIN_SAMPLES_PER_PARAMETER,
        adapt_n_states: bool = True,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if n_init < 1:
            raise ValueError("n_init must be >= 1")
        if covariance_type not in _COVARIANCE_TYPES:
            raise ValueError(
                f"Unknown covariance_type {covariance_type!r}; expected one of {_COVARIANCE_TYPES}"
            )
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_init = n_init
        self.min_covar = min_covar
        self.tol = tol
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.min_samples_per_param = min_samples_per_param
        self.adapt_n_states = adapt_n_states

        self.model: hmm.GaussianHMM | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = list(HMM_FEATURES)
        self.fitted: bool = False

        # Learned parameters (canonical state order: ascending volatility)
        self.transition_matrix: np.ndarray | None = None
        self.state_means: np.ndarray | None = None
        self.state_covariances: np.ndarray | None = None
        self.state_summaries: list[StateSummary] = []
        self.state_regimes: dict[int, MarketRegime] = {}
        self.training_score: float | None = None
        self._fit_info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Features and sizing
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Causal, stationary HMM features (see :func:`build_hmm_features`)."""
        features = build_hmm_features(df)
        self.feature_names = list(features.columns)
        return features

    @property
    def n_parameters(self) -> int:
        """Free parameters of this model configuration."""
        return n_free_parameters(self.n_states, len(HMM_FEATURES), self.covariance_type)

    @property
    def min_training_rows(self) -> int:
        """Feature rows needed to fit (``min_samples_per_param`` per free parameter)."""
        return math.ceil(self.min_samples_per_param * self.n_parameters)

    @property
    def min_training_bars(self) -> int:
        """OHLCV bars needed to fit, including the feature warm-up."""
        return self.min_training_rows + HMM_WARMUP_BARS

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> TrueHMMDetector:
        """
        Fit the HMM with Baum-Welch (EM), keeping the best of ``n_init`` restarts.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Self for method chaining

        Raises:
            ValueError: If there is too little data for the model size, or
                every EM restart fails
        """
        X = self._prepare_features(df)
        if len(X) < self.min_training_rows and self.adapt_n_states:
            self._reduce_n_states(len(X))
        if len(X) < self.min_training_rows:
            raise ValueError(
                f"Insufficient data for a {self.n_states}-state '{self.covariance_type}' HMM: "
                f"{self.n_parameters} free parameters need at least {self.min_training_rows} "
                f"feature rows ({self.min_samples_per_param:g} per parameter, i.e. "
                f"{self.min_training_bars} bars including the {HMM_WARMUP_BARS}-bar warm-up); "
                f"got {len(X)} rows. Load more history or use fewer states."
            )

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            best: hmm.GaussianHMM | None = None
            best_score = -np.inf
            best_seed: int | None = None
            errors: list[str] = []
            for k in range(self.n_init):
                seed = None if self.random_state is None else self.random_state + k
                model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    min_covar=self.min_covar,
                    random_state=seed,
                )
                try:
                    with _quiet_hmmlearn():
                        model.fit(X_scaled)
                        score = float(model.score(X_scaled))
                except Exception as exc:  # one bad restart must not sink the fit
                    errors.append(str(exc))
                    logger.debug("HMM restart with seed %s failed: %s", seed, exc)
                    continue
                if np.isfinite(score) and score > best_score:
                    best, best_score, best_seed = model, score, seed

            if best is None:
                raise ValueError(f"all {self.n_init} EM restarts failed: {errors[:3]}")

            self._check_convergence(best, best_seed)
            self._canonicalize_states(best)
        except Exception as e:
            raise ValueError(f"HMM fitting failed: {e!s}") from e

        self.model = best
        self.scaler = scaler
        self.transition_matrix = np.asarray(best.transmat_)
        self.state_means = np.asarray(best.means_)
        self.state_covariances = np.asarray(best.covars_)
        self.training_score = best_score
        self.state_summaries = [
            self._summarize_state(i, best, scaler) for i in range(self.n_states)
        ]
        self.state_regimes = {
            i: classify_state(s, self.thresholds) for i, s in enumerate(self.state_summaries)
        }
        self.fitted = True
        return self

    def _reduce_n_states(self, n_rows: int) -> None:
        """Shrink ``n_states`` to the largest count (>= 2) that ``n_rows`` feature rows support."""
        requested = self.n_states
        n = requested
        while n > 2 and n_rows < math.ceil(
            self.min_samples_per_param
            * n_free_parameters(n, len(HMM_FEATURES), self.covariance_type)
        ):
            n -= 1
        if n != requested:
            logger.warning(
                "Only %d feature rows: fitting a %d-state HMM instead of %d states "
                "(%g rows per free parameter required)",
                n_rows,
                n,
                requested,
                self.min_samples_per_param,
            )
            self.n_states = n

    def _check_convergence(self, model: hmm.GaussianHMM, seed: int | None) -> None:
        """Log (warning) EM log-likelihood decreases and non-convergence of the chosen model."""
        monitor = model.monitor_
        history = np.asarray(list(monitor.history), dtype=float)
        deltas = np.diff(history)
        worst_drop = float(-deltas.min()) if len(deltas) else 0.0
        last_gain = float(deltas[-1]) if len(deltas) else 0.0
        # hmmlearn also stops early on a likelihood *decrease*; that is not convergence
        converged = last_gain >= 0 and (monitor.iter < self.n_iter or last_gain < self.tol)
        self._fit_info = {
            "best_seed": seed,
            "n_iterations": int(monitor.iter),
            "converged": converged,
            "max_loglik_decrease": max(worst_drop, 0.0),
        }
        if worst_drop > self.tol:
            logger.warning(
                "HMM EM log-likelihood decreased by %.4g during fitting (seed %s); "
                "the fitted model may be degenerate",
                worst_drop,
                seed,
            )
        if not converged:
            logger.warning(
                "HMM EM did not converge within %d iterations (seed %s, tol %g)",
                self.n_iter,
                seed,
                self.tol,
            )

    def _canonicalize_states(self, model: hmm.GaussianHMM) -> None:
        """
        Reorder states in place by ascending volatility (ties: trend strength).

        HMM state indices are otherwise arbitrary and change with the seed;
        a canonical order keeps state ints and their labels stable across refits.
        """
        vol_idx = self.feature_names.index("log_volatility")
        trend_idx = self.feature_names.index("trend_strength")
        means = np.asarray(model.means_)
        order = np.lexsort((means[:, trend_idx], means[:, vol_idx]))
        if np.array_equal(order, np.arange(self.n_states)):
            return
        model.startprob_ = np.asarray(model.startprob_)[order]
        model.transmat_ = np.asarray(model.transmat_)[order][:, order]
        model.means_ = means[order]
        if self.covariance_type != "tied":
            # hmmlearn keeps covariances in their compact per-type form in _covars_
            model._covars_ = np.asarray(model._covars_)[order]

    def _summarize_state(
        self, state: int, model: hmm.GaussianHMM, scaler: StandardScaler
    ) -> StateSummary:
        """De-standardize a state's emission mean into interpretable units."""
        raw = scaler.inverse_transform(np.asarray(model.means_)[state : state + 1])[0]
        values = dict(zip(self.feature_names, (float(v) for v in raw), strict=True))
        vol_idx = self.feature_names.index("log_volatility")
        typical_log_vol = float(scaler.mean_[vol_idx])
        state_vol = math.exp(values["log_volatility"])
        return StateSummary(
            rel_vol=math.exp(values["log_volatility"] - typical_log_vol),
            trend_score=values["trend_strength"] / state_vol if state_vol > 0 else math.nan,
            vol_expansion=values["vol_expansion"],
            autocorr=values["autocorr_1"],
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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

    def _filtered(self, df: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
        """Feature-row index and filtered posteriors for ``df``."""
        model, scaler, _ = self._require_fitted()
        features = build_hmm_features(df)
        with _quiet_hmmlearn():
            posteriors = forward_filter(model, scaler.transform(features))
        return features.index, posteriors

    def filtered_posteriors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtered state posteriors ``P(s_t | o_1..o_t)`` for every feature row.

        Forward-only: the row for bar ``t`` never depends on bars after ``t``,
        so this is safe for regime *history* (charts, backtests, persistence).

        Returns:
            DataFrame indexed like the feature rows, one column per state
        """
        index, posteriors = self._filtered(df)
        return pd.DataFrame(posteriors, index=index, columns=list(range(self.n_states)))

    def regime_history(self, df: pd.DataFrame) -> pd.Series:
        """Causal per-bar regime (argmax of the filtered posterior), indexed by feature row."""
        index, posteriors = self._filtered(df)
        states = np.argmax(posteriors, axis=1)
        return pd.Series([self.state_regimes[int(s)] for s in states], index=index)

    def _current_posterior(self, df: pd.DataFrame) -> np.ndarray:
        """Filtered posterior state distribution for the most recent bar."""
        _, posteriors = self._filtered(df)
        current: np.ndarray = posteriors[-1]
        return current

    def predict_with_states(self, df: pd.DataFrame) -> tuple[MarketRegime, np.ndarray, float]:
        """
        Predict the current regime and the causal state history in one pass.

        Returns:
            Tuple of (regime, states, confidence): ``states`` holds the argmax
            of the filtered posterior for each feature row (the last element
            is the current state) and ``confidence`` its posterior probability.

        Raises:
            ValueError: If the model is not fitted or prediction fails
        """
        self._require_fitted()
        try:
            _, posteriors = self._filtered(df)
        except Exception as e:
            raise ValueError(f"Prediction failed: {e!s}") from e
        states = np.argmax(posteriors, axis=1).astype(int)
        current = int(states[-1])
        return self.state_regimes[current], states, float(posteriors[-1, current])

    def predict_regime(
        self, df: pd.DataFrame, use_viterbi: bool = False
    ) -> tuple[MarketRegime, int, float]:
        """
        Predict the regime of the last bar of ``df``.

        Args:
            df: DataFrame with OHLCV data
            use_viterbi: If False (default), take the argmax of the filtered
                posterior at the last bar -- the same state the analyzer and
                walk-forward validation use. If True, take the last state of
                the Viterbi path (most likely joint sequence), which can differ.

        Returns:
            Tuple of (regime, state, confidence); ``confidence`` is the
            filtered posterior probability of the returned state.

        Raises:
            ValueError: If model not fitted or prediction fails
        """
        model, scaler, _ = self._require_fitted()
        try:
            X_scaled = scaler.transform(build_hmm_features(df))
            with _quiet_hmmlearn():
                posterior = forward_filter(model, X_scaled)[-1]
                state = int(model.predict(X_scaled)[-1]) if use_viterbi else int(posterior.argmax())
        except Exception as e:
            raise ValueError(f"Prediction failed: {e!s}") from e
        return self.state_regimes[state], state, float(posterior[state])

    # ------------------------------------------------------------------
    # Model summaries
    # ------------------------------------------------------------------

    def calculate_regime_persistence(self, states: np.ndarray, lookback: int = 20) -> float:
        """Share of the last ``lookback`` states equal to the current one."""
        return regime_persistence(states, lookback)

    def get_transition_probability(self, from_state: int, to_state: int) -> float:
        """
        Learned one-bar transition probability between states.

        Raises:
            ValueError: If the model is not fitted or a state index is invalid
        """
        return transition_probability(
            self.transition_matrix if self.fitted else None, from_state, to_state
        )

    def get_training_convergence(self) -> dict:
        """
        HMM training diagnostics.

        Returns:
            Dict with the total training ``log_likelihood``, ``converged``
            (our own check: EM stopped before ``n_iter`` or the last gain was
            below ``tol`` -- hmmlearn reports ``converged=True`` whenever it
            hits ``n_iter``), ``n_iterations``, ``best_seed``,
            ``max_loglik_decrease`` and model-size fields
        """
        if not self.fitted or self.model is None:
            return {"fitted": False}

        return {
            "fitted": True,
            "log_likelihood": self.training_score,
            "n_states": self.n_states,
            "n_features": len(self.feature_names),
            "n_parameters": self.n_parameters,
            "covariance_type": self.covariance_type,
            "n_init": self.n_init,
            **self._fit_info,
        }

    def _map_state_index_to_regime(self, state_index: int) -> MarketRegime:
        """Regime label of ``state_index`` (UNKNOWN if unfitted or out of range)."""
        return self.state_regimes.get(state_index, MarketRegime.UNKNOWN)

    def get_state_regime_map(self) -> dict[int, MarketRegime]:
        """
        Mapping from every state index to its MarketRegime.

        Raises:
            ValueError: If model not fitted
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting state map")
        return dict(self.state_regimes)

    def forecast_regime_probabilities(self, df: pd.DataFrame, n_steps: int = 1) -> np.ndarray:
        """
        Forecast the state distribution ``n_steps`` ahead: ``pi_{t+n} = pi_t @ T^n``.

        ``pi_t`` is the filtered posterior at the last bar of ``df``.

        Raises:
            ValueError: If model not fitted or n_steps < 1
        """
        _, _, transmat = self._require_fitted()
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")

        pi_t = self._current_posterior(df)
        forecast: np.ndarray = pi_t @ np.linalg.matrix_power(transmat, n_steps)
        return forecast

    def forecast_regime_sequence(self, df: pd.DataFrame, n_steps: int = 5) -> list[dict]:
        """
        Forecast regime probabilities for each step from 1 to n_steps.

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
        state_regime_map = self.get_state_regime_map()

        results = []
        for step in range(1, n_steps + 1):
            forecast_probs = pi_t @ np.linalg.matrix_power(transmat, step)

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
        Regime stability metrics from the learned transition matrix.

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
        self_trans = {i: float(T[i, i]) for i in range(self.n_states)}
        expected_dur = {
            i: 1.0 / (1.0 - T[i, i]) if T[i, i] < 1.0 else float("inf")
            for i in range(self.n_states)
        }

        # Stationary distribution: left eigenvector of T for eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(T.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()

        state_regime_map = self.get_state_regime_map()
        stationary_regimes: dict[MarketRegime, float] = {}
        for state_idx, prob in enumerate(stationary):
            regime = state_regime_map[state_idx]
            stationary_regimes[regime] = stationary_regimes.get(regime, 0.0) + float(prob)

        return {
            "self_transition_probs": self_trans,
            "expected_durations": expected_dur,
            "stationary_distribution": stationary,
            "stationary_regimes": stationary_regimes,
        }

    def compare_with_gmm(self, df: pd.DataFrame, gmm_detector: Any) -> dict:
        """
        Compare the current prediction with another detector's.

        .. deprecated::
            The GMM detector has been folded into this class, so there is no
            second model to compare against. Kept for API compatibility.
        """
        warnings.warn(
            "TrueHMMDetector.compare_with_gmm is deprecated: the GMM detector was "
            "removed and HiddenMarkovRegimeDetector is now an alias of TrueHMMDetector.",
            DeprecationWarning,
            stacklevel=2,
        )
        hmm_regime, hmm_state, hmm_confidence = self.predict_regime(df)
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


def select_n_states(
    df: pd.DataFrame,
    candidates: Iterable[int] = range(2, 7),
    **detector_kwargs: Any,
) -> tuple[int, dict[int, float]]:
    """
    Choose the number of HMM states by the Bayesian information criterion.

    Fits a :class:`TrueHMMDetector` for every candidate the data can support
    (see ``min_samples_per_param``) and returns the one with the lowest BIC.

    Args:
        df: OHLCV training data
        candidates: State counts to try
        **detector_kwargs: Forwarded to :class:`TrueHMMDetector` (``n_states`` is ignored)

    Returns:
        Tuple of (best n_states, {n_states: BIC} for every candidate that fit)

    Raises:
        ValueError: If no candidate could be fitted
    """
    detector_kwargs.pop("n_states", None)
    # each candidate must be fitted at its own size, never silently shrunk
    detector_kwargs["adapt_n_states"] = False
    scores: dict[int, float] = {}
    for n in candidates:
        detector = TrueHMMDetector(n_states=n, **detector_kwargs)
        try:
            detector.fit(df)
        except ValueError as exc:
            logger.info("Skipping n_states=%d in BIC selection: %s", n, exc)
            continue
        model, scaler, _ = detector._require_fitted()
        X_scaled = scaler.transform(detector._prepare_features(df))
        scores[n] = float(model.bic(X_scaled))
    if not scores:
        raise ValueError("No candidate state count could be fitted on this data")
    best = min(scores, key=lambda n: scores[n])
    return best, scores
