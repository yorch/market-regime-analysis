"""
Deprecated GMM-era detector name.

``HiddenMarkovRegimeDetector`` used to be a Gaussian-mixture approximation of
an HMM (independent per-bar clustering plus a post-hoc transition matrix) and
was the analyzer's model, while walk-forward validation used
:class:`~mra_lib.indicators.true_hmm_detector.TrueHMMDetector`. There is now a
single model: this name is a thin deprecated alias of ``TrueHMMDetector`` and
will be removed in a future release.
"""

from __future__ import annotations

import warnings
from typing import Any

from .true_hmm_detector import TrueHMMDetector


class HiddenMarkovRegimeDetector(TrueHMMDetector):
    """
    Deprecated alias of :class:`TrueHMMDetector`.

    .. deprecated::
        Use :class:`~mra_lib.indicators.true_hmm_detector.TrueHMMDetector`.
        Constructing this class emits a :class:`DeprecationWarning`; it accepts
        the same arguments and behaves identically.
    """

    def __init__(self, n_states: int = 6, **kwargs: Any) -> None:
        warnings.warn(
            "HiddenMarkovRegimeDetector (GMM) is deprecated and is now an alias of "
            "TrueHMMDetector; use mra_lib.indicators.TrueHMMDetector instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(n_states=n_states, **kwargs)
