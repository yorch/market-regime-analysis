"""
Market Regime Analysis System

A professional-grade Python application that implements Jim Simons' Hidden
Markov Model methodology for market regime detection and quantitative trading
analysis, following the exact approach used by Renaissance Technologies.
"""

from .analyzer import MarketRegimeAnalyzer
from .data_classes import RegimeAnalysis
from .enums import MarketRegime, TradingStrategy
from .hmm_detector import HiddenMarkovRegimeDetector
from .portfolio import PortfolioHMMAnalyzer
from .risk_calculator import SimonsRiskCalculator

__version__ = "1.0.0"
__all__ = [
    "HiddenMarkovRegimeDetector",
    "MarketRegime",
    "MarketRegimeAnalyzer",
    "PortfolioHMMAnalyzer",
    "RegimeAnalysis",
    "SimonsRiskCalculator",
    "TradingStrategy",
]
