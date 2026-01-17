"""
Test and compare True HMM implementation vs GMM approach.

This script validates the new TrueHMMDetector and compares it
with the original GMM-based HiddenMarkovRegimeDetector.
"""

import sys

import pandas as pd

from market_regime_analysis.hmm_detector import HiddenMarkovRegimeDetector
from market_regime_analysis.providers import MarketDataProvider
from market_regime_analysis.true_hmm_detector import TrueHMMDetector


def main():
    """Run comparison test between True HMM and GMM."""
    print("=" * 100)
    print("COMPARING TRUE HMM vs GMM APPROACH")
    print("=" * 100)

    # Load data
    print("\n1. Loading market data...")
    try:
        provider = MarketDataProvider.create_provider("yfinance")
        df = provider.fetch("SPY", "2y", "1d")
        print(f"   ✓ Loaded {len(df)} days of SPY data")
    except Exception as e:
        print(f"   ✗ Failed to load data: {e}")
        sys.exit(1)

    # Test True HMM
    print("\n2. Testing True HMM (hmmlearn-based)...")
    try:
        true_hmm = TrueHMMDetector(n_states=6, n_iter=100)
        true_hmm.fit(df)

        convergence = true_hmm.get_training_convergence()
        print(f"   ✓ HMM trained successfully")
        print(f"     - Log-likelihood: {convergence['log_likelihood']:.2f}")
        print(f"     - Converged: {convergence['converged']}")
        print(f"     - Features: {convergence['n_features']}")

        # Predict regime
        regime, state, confidence = true_hmm.predict_regime(df, use_viterbi=True)
        print(f"\n   True HMM Prediction:")
        print(f"     - Regime: {regime.value}")
        print(f"     - State: {state}")
        print(f"     - Confidence: {confidence:.2%}")

        # Show transition matrix
        print(f"\n   Learned Transition Matrix (top 3 transitions):")
        for i in range(min(3, true_hmm.n_states)):
            top_transitions = sorted(
                [(j, true_hmm.transition_matrix[i, j]) for j in range(true_hmm.n_states)],
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            print(
                f"     State {i} -> {', '.join([f'{j}({p:.2%})' for j, p in top_transitions])}"
            )

    except Exception as e:
        print(f"   ✗ True HMM failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test GMM approach
    print("\n3. Testing GMM Approach (original)...")
    try:
        gmm_detector = HiddenMarkovRegimeDetector(n_states=6)
        gmm_detector.fit(df)
        print(f"   ✓ GMM trained successfully")

        # Predict regime
        regime, state, confidence = gmm_detector.predict_regime(df)
        print(f"\n   GMM Prediction:")
        print(f"     - Regime: {regime.value}")
        print(f"     - State: {state}")
        print(f"     - Confidence: {confidence:.2%}")

    except Exception as e:
        print(f"   ✗ GMM failed: {e}")
        import traceback

        traceback.print_exc()

    # Compare approaches
    print("\n4. Comparing Approaches...")
    try:
        comparison = true_hmm.compare_with_gmm(df, gmm_detector)
        print(f"\n   Comparison Results:")
        print(f"     - Regime Agreement: {comparison['regime_agreement']}")
        print(f"     - HMM Regime: {comparison['hmm_regime']}")
        print(f"     - GMM Regime: {comparison['gmm_regime']}")
        print(f"     - HMM Confidence: {comparison['hmm_confidence']:.2%}")
        print(f"     - GMM Confidence: {comparison['gmm_confidence']:.2%}")

    except Exception as e:
        print(f"   ✗ Comparison failed: {e}")

    # Key differences explanation
    print("\n5. Key Methodological Differences:")
    print("\n   GMM Approach (Original):")
    print("     ✗ Treats each observation independently")
    print("     ✗ No temporal modeling")
    print("     ✗ Transition matrix calculated post-hoc")
    print("     ✗ States assigned by clustering, not dynamics")

    print("\n   True HMM (New):")
    print("     ✓ Models temporal dependencies explicitly")
    print("     ✓ Uses Baum-Welch for parameter learning")
    print("     ✓ Uses Viterbi for optimal state sequences")
    print("     ✓ Transition matrix learned during training")
    print("     ✓ State sequences respect temporal dynamics")

    print("\n" + "=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)

    print("\nNext Steps:")
    print("1. Integrate TrueHMMDetector into MarketRegimeAnalyzer")
    print("2. Run backtests to compare performance")
    print("3. Update documentation with methodology comparison")
    print("4. Add configuration option to switch between implementations")


if __name__ == "__main__":
    main()
