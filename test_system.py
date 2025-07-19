#!/usr/bin/env python3
"""
Simple test script to verify the market regime analysis system works correctly.
"""

import os
import sys

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_regime_analysis import MarketRegimeAnalyzer


def test_basic_functionality():
    """Test basic functionality of the system."""
    print("Testing Market Regime Analysis System...")

    try:
        # Test 1: Basic analyzer initialization
        print("\n1. Testing analyzer initialization...")
        analyzer = MarketRegimeAnalyzer("SPY", periods={"1D": "1y"})
        print("✓ Analyzer initialized successfully")

        # Test 2: Regime analysis
        print("\n2. Testing regime analysis...")
        analysis = analyzer.analyze_current_regime("1D")
        print(f"✓ Current regime: {analysis.current_regime.value}")
        print(f"✓ Confidence: {analysis.regime_confidence:.2f}")
        print(f"✓ Strategy: {analysis.recommended_strategy.value}")

        # Test 3: Print report
        print("\n3. Testing report generation...")
        analyzer.print_analysis_report("1D")
        print("✓ Report generated successfully")

        print("\n🎉 All tests passed! The system is working correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e!s}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
