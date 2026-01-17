#!/usr/bin/env python3
"""
Simple test script with mock data to verify core functionality.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_mock_data() -> pd.DataFrame:
    """Create mock OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")

    # Generate realistic price data
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [100.0]  # Starting price

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data
    df = pd.DataFrame(index=dates)
    df["Close"] = prices
    df["Open"] = df["Close"].shift(1) * (1 + np.random.normal(0, 0.005, len(df)))
    df["High"] = np.maximum(df["Open"], df["Close"]) * (
        1 + np.abs(np.random.normal(0, 0.01, len(df)))
    )
    df["Low"] = np.minimum(df["Open"], df["Close"]) * (
        1 - np.abs(np.random.normal(0, 0.01, len(df)))
    )
    df["Volume"] = np.random.randint(1000000, 10000000, len(df))

    # Fill any NaN values (using pandas 2.0+ syntax)
    df = df.ffill().bfill()

    return df


def test_core_functionality():
    """Test core functionality with mock data."""
    print("Testing Market Regime Analysis System with Mock Data...")

    try:
        # Import after setting path
        from market_regime_analysis.hmm_detector import HiddenMarkovRegimeDetector
        from market_regime_analysis.risk_calculator import SimonsRiskCalculator

        # Test 1: Create mock data
        print("\n1. Creating mock data...")
        mock_data = create_mock_data()
        print(f"✓ Created {len(mock_data)} days of mock data")

        # Test 2: HMM Detector
        print("\n2. Testing HMM detector...")
        hmm = HiddenMarkovRegimeDetector(n_states=6)
        hmm.fit(mock_data)
        print("✓ HMM fitted successfully")

        # Test 3: Regime prediction
        print("\n3. Testing regime prediction...")
        regime, state, confidence = hmm.predict_regime(mock_data)
        print(f"✓ Predicted regime: {regime.value}")
        print(f"✓ HMM state: {state}")
        print(f"✓ Confidence: {confidence:.3f}")

        # Test 4: Risk calculator
        print("\n4. Testing risk calculator...")
        kelly_size = SimonsRiskCalculator.calculate_kelly_optimal_size(
            win_rate=0.55, avg_win=0.02, avg_loss=0.015, confidence=0.8
        )

        regime_size = SimonsRiskCalculator.calculate_regime_adjusted_size(
            base_size=0.02, regime=regime, confidence=confidence, persistence=0.7
        )

        print(f"✓ Kelly optimal size: {kelly_size:.3f}")
        print(f"✓ Regime adjusted size: {regime_size:.3f}")

        # Test 5: Feature engineering (from HMM detector)
        print("\n5. Testing feature engineering...")
        features = hmm._prepare_features(mock_data)
        print(f"✓ Generated {len(features.columns)} features")
        print(f"✓ Feature names: {', '.join(features.columns[:5])}...")

        print("\n🎉 All core functionality tests passed!")
        print(f"✓ System is working correctly with {len(mock_data)} data points")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e!s}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_core_functionality()
    sys.exit(0 if success else 1)
