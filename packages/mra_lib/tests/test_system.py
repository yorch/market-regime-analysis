"""End-to-end smoke test against a live data provider (requires network access)."""

import pytest

from mra_lib import MarketRegimeAnalyzer
from mra_lib.config.enums import MarketRegime, TradingStrategy


@pytest.mark.integration
def test_analyzer_end_to_end_with_live_yfinance_data():
    """The analyzer fetches live SPY data and produces a well-formed daily analysis."""
    analyzer = MarketRegimeAnalyzer("SPY", periods={"1D": "1y"}, provider_flag="yfinance")

    analysis = analyzer.analyze_current_regime("1D")

    assert isinstance(analysis.current_regime, MarketRegime)
    assert isinstance(analysis.recommended_strategy, TradingStrategy)
    assert 0.0 <= analysis.regime_confidence <= 1.0
