"""
Market Regime Analysis Library (mra_lib)

Core library for Hidden Markov Model (HMM) market regime detection
and quantitative trading analysis.
Zero UI/framework dependencies — designed for embedding in CLIs, web apps, bots, etc.

The library never prints. Progress and diagnostics go to the standard
``logging`` module under the ``mra_lib`` logger hierarchy (a ``NullHandler`` is
installed, so nothing is emitted unless the application configures logging).
Human-readable reports are returned as strings by ``format_*`` methods.
Deliberate failures raise subclasses of :class:`mra_lib.errors.MRAError`.
"""

import logging
from importlib.metadata import PackageNotFoundError, version as _dist_version

from .analyzer import MarketRegimeAnalyzer
from .config.data_classes import RegimeAnalysis
from .config.enums import MarketRegime, TradingStrategy
from .errors import (
    DataLoadError,
    InsufficientDataError,
    ModelNotFittedError,
    MRAError,
    ProviderError,
)
from .indicators.hmm_detector import HiddenMarkovRegimeDetector
from .indicators.true_hmm_detector import TrueHMMDetector
from .portfolio.portfolio import PortfolioHMMAnalyzer
from .risk.risk_calculator import PortfolioPositionLimits, PositionRecord, SimonsRiskCalculator

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = _dist_version("mra-lib")
except PackageNotFoundError:  # pragma: no cover - not installed (e.g. bare source checkout)
    __version__ = "0.0.0+unknown"

__all__ = [
    "DataLoadError",
    "HiddenMarkovRegimeDetector",
    "InsufficientDataError",
    "MRAError",
    "MarketRegime",
    "MarketRegimeAnalyzer",
    "ModelNotFittedError",
    "PortfolioHMMAnalyzer",
    "PortfolioPositionLimits",
    "PositionRecord",
    "ProviderError",
    "RegimeAnalysis",
    "SimonsRiskCalculator",
    "TradingStrategy",
    "TrueHMMDetector",
]
