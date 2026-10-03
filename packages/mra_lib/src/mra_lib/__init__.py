"""
Market Regime Analysis Library (mra_lib)

Core library implementing Jim Simons' Hidden Markov Model methodology
for market regime detection and quantitative trading analysis.
Zero UI/framework dependencies — designed for embedding in CLIs, web apps, bots, etc.
"""

from importlib.metadata import PackageNotFoundError, version as _dist_version

from .analyzer import MarketRegimeAnalyzer
from .config.data_classes import RegimeAnalysis
from .config.enums import MarketRegime, TradingStrategy
from .indicators.hmm_detector import HiddenMarkovRegimeDetector
from .indicators.true_hmm_detector import TrueHMMDetector
from .portfolio.portfolio import PortfolioHMMAnalyzer
from .risk.risk_calculator import PortfolioPositionLimits, PositionRecord, SimonsRiskCalculator

try:
    __version__ = _dist_version("mra-lib")
except PackageNotFoundError:  # pragma: no cover - not installed (e.g. bare source checkout)
    __version__ = "0.0.0+unknown"

__all__ = [
    "HiddenMarkovRegimeDetector",
    "MarketRegime",
    "MarketRegimeAnalyzer",
    "PortfolioHMMAnalyzer",
    "PortfolioPositionLimits",
    "PositionRecord",
    "RegimeAnalysis",
    "SimonsRiskCalculator",
    "TradingStrategy",
    "TrueHMMDetector",
]
