"""HMM-based market regime detection: features, detector, and state-to-regime mapping."""

from .base import RegimeDetector, regime_persistence, transition_probability
from .features import HMM_FEATURES, build_hmm_features
from .hmm_detector import HiddenMarkovRegimeDetector
from .regime_mapping import RegimeThresholds, classify_state
from .true_hmm_detector import TrueHMMDetector, select_n_states

__all__ = [
    "HMM_FEATURES",
    "HiddenMarkovRegimeDetector",
    "RegimeDetector",
    "RegimeThresholds",
    "TrueHMMDetector",
    "build_hmm_features",
    "classify_state",
    "regime_persistence",
    "select_n_states",
    "transition_probability",
]
