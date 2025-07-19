"""
Portfolio HMM analyzer for multi-asset regime analysis.

This module implements portfolio-level analysis following Renaissance
Technologies' approach to multi-asset regime detection and correlation analysis.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from .analyzer import MarketRegimeAnalyzer
from .enums import MarketRegime


class PortfolioHMMAnalyzer:
    """
    Multi-asset regime analysis following Renaissance approach.

    This class extends the single-asset analysis to portfolio-level
    regime detection, correlation analysis, and statistical arbitrage
    pair identification across multiple assets.
    """

    def __init__(
        self, symbols: list[str], periods: dict[str, str] | None = None
    ) -> None:
        """
        Initialize portfolio analyzer.

        Args:
            symbols: List of trading symbols to analyze
            periods: Dictionary mapping timeframes to data periods
        """
        self.symbols = symbols
        self.periods = periods
        self.analyzers: dict[str, MarketRegimeAnalyzer] = {}
        self.portfolio_data: dict[str, pd.DataFrame] = {}

        print(f"Initializing portfolio analysis for {len(symbols)} symbols...")

        # Initialize individual analyzers
        for symbol in symbols:
            try:
                analyzer = MarketRegimeAnalyzer(symbol, periods)
                self.analyzers[symbol] = analyzer
                print(f"✓ Initialized {symbol}")
            except Exception as e:
                print(f"✗ Failed to initialize {symbol}: {e!s}")

        self._prepare_portfolio_data()

    def _prepare_portfolio_data(self) -> None:
        """Prepare aligned portfolio data for correlation analysis."""
        print("Preparing portfolio correlation data...")

        for timeframe in self.periods or {"1D": "2y"}:
            price_data = {}

            for symbol, analyzer in self.analyzers.items():
                if timeframe in analyzer.data:
                    price_data[symbol] = analyzer.data[timeframe]["Close"]

            if price_data:
                # Align all price series
                portfolio_df = pd.DataFrame(price_data)
                portfolio_df = portfolio_df.dropna()

                # Calculate returns and correlations
                returns = portfolio_df.pct_change().dropna()
                portfolio_df["portfolio_return"] = returns.mean(axis=1)
                portfolio_df["portfolio_volatility"] = returns.std(axis=1)

                self.portfolio_data[timeframe] = portfolio_df
                print(
                    f"✓ Prepared {timeframe} portfolio data: {len(portfolio_df)} periods"
                )

    def calculate_regime_correlations(self, timeframe: str = "1D") -> pd.DataFrame:
        """
        Calculate cross-asset regime correlations.

        Args:
            timeframe: Timeframe for correlation analysis

        Returns:
            DataFrame with regime correlation matrix
        """
        if timeframe not in self.portfolio_data:
            raise ValueError(f"Portfolio data not available for {timeframe}")

        # Get regime predictions for all symbols
        regime_data = {}

        for symbol, analyzer in self.analyzers.items():
            try:
                analysis = analyzer.analyze_current_regime(timeframe)
                regime_data[symbol] = {
                    "regime": analysis.current_regime.value,
                    "confidence": analysis.regime_confidence,
                    "state": analysis.hmm_state,
                    "persistence": analysis.regime_persistence,
                }
            except Exception as e:
                print(f"Error analyzing {symbol}: {e!s}")
                continue

        # Create correlation matrix
        regime_df = pd.DataFrame(regime_data).T

        # Calculate price correlations
        price_data = self.portfolio_data[timeframe][list(regime_data.keys())]
        if len(price_data.columns) > 1:
            price_correlations = price_data.corr()
            regime_df = regime_df.join(price_correlations.add_suffix("_price_corr"))

        return regime_df

    def get_portfolio_regime_summary(self, timeframe: str = "1D") -> dict[str, any]:
        """
        Get portfolio-level regime metrics.

        Args:
            timeframe: Timeframe for analysis

        Returns:
            Dictionary with portfolio metrics
        """
        summary = {
            "dominant_regime": None,
            "regime_consensus": 0.0,
            "average_confidence": 0.0,
            "risk_level": "Unknown",
            "diversification_benefit": 0.0,
            "regime_distribution": {},
            "correlation_risk": 0.0,
        }

        try:
            # Get all regime analyses
            analyses = []
            for symbol, analyzer in self.analyzers.items():
                try:
                    analysis = analyzer.analyze_current_regime(timeframe)
                    analyses.append((symbol, analysis))
                except Exception:
                    continue

            if not analyses:
                return summary

            # Calculate regime distribution
            regimes = [analysis.current_regime for _, analysis in analyses]
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1

            summary["regime_distribution"] = regime_counts

            # Find dominant regime
            if regime_counts:
                dominant_regime = max(regime_counts.items(), key=lambda x: x[1])
                summary["dominant_regime"] = dominant_regime[0]
                summary["regime_consensus"] = dominant_regime[1] / len(analyses)

            # Calculate average confidence
            confidences = [analysis.regime_confidence for _, analysis in analyses]
            summary["average_confidence"] = np.mean(confidences)

            # Assess portfolio risk level
            high_vol_count = sum(
                1
                for _, analysis in analyses
                if analysis.current_regime == MarketRegime.HIGH_VOLATILITY
            )
            unknown_count = sum(
                1
                for _, analysis in analyses
                if analysis.current_regime == MarketRegime.UNKNOWN
            )

            total_assets = len(analyses)
            if (high_vol_count + unknown_count) / total_assets > 0.5:
                summary["risk_level"] = "High"
            elif (high_vol_count + unknown_count) / total_assets > 0.3:
                summary["risk_level"] = "Medium"
            else:
                summary["risk_level"] = "Low"

            # Calculate correlation risk if we have portfolio data
            if (
                timeframe in self.portfolio_data
                and len(self.portfolio_data[timeframe].columns) > 2
            ):
                symbols_in_data = [
                    s
                    for s, _ in analyses
                    if s in self.portfolio_data[timeframe].columns
                ]
                if len(symbols_in_data) > 1:
                    price_data = self.portfolio_data[timeframe][symbols_in_data]
                    corr_matrix = price_data.corr()

                    # Average absolute correlation (excluding diagonal)
                    n = len(corr_matrix)
                    total_corr = corr_matrix.abs().sum().sum() - n  # Exclude diagonal
                    avg_corr = total_corr / (n * (n - 1))
                    summary["correlation_risk"] = avg_corr

                    # Diversification benefit (lower correlation = higher benefit)
                    summary["diversification_benefit"] = 1.0 - avg_corr

        except Exception as e:
            print(f"Error calculating portfolio summary: {e!s}")

        return summary

    def identify_arbitrage_pairs(self, timeframe: str = "1D") -> list[dict[str, any]]:
        """
        Statistical arbitrage pairs detection.

        Args:
            timeframe: Timeframe for analysis

        Returns:
            List of arbitrage opportunities
        """
        opportunities = []

        if timeframe not in self.portfolio_data:
            return opportunities

        # Get available symbols with data
        available_symbols = [s for s in self.symbols if s in self.analyzers]

        if len(available_symbols) < 2:
            return opportunities

        # Analyze all possible pairs
        for symbol1, symbol2 in combinations(available_symbols, 2):
            try:
                # Get price data for both symbols
                if (
                    symbol1 not in self.portfolio_data[timeframe].columns
                    or symbol2 not in self.portfolio_data[timeframe].columns
                ):
                    continue

                price1 = self.portfolio_data[timeframe][symbol1]
                price2 = self.portfolio_data[timeframe][symbol2]

                # Calculate correlation
                correlation = price1.corr(price2)

                # Skip if correlation is too low
                if abs(correlation) < 0.3:
                    continue

                # Calculate spread
                returns1 = price1.pct_change().dropna()
                returns2 = price2.pct_change().dropna()

                # Align returns
                common_index = returns1.index.intersection(returns2.index)
                if len(common_index) < 50:
                    continue

                aligned_returns1 = returns1.loc[common_index]
                aligned_returns2 = returns2.loc[common_index]

                # Calculate spread Z-score
                spread = aligned_returns1 - aligned_returns2
                spread_mean = spread.rolling(50).mean()
                spread_std = spread.rolling(50).std()
                spread_zscore = (spread - spread_mean) / (spread_std + 1e-8)

                current_zscore = spread_zscore.iloc[-1]

                # Identify opportunity
                if abs(current_zscore) > 2.0 and not pd.isna(current_zscore):
                    # Get regime analyses for both symbols
                    try:
                        analysis1 = self.analyzers[symbol1].analyze_current_regime(
                            timeframe
                        )
                        analysis2 = self.analyzers[symbol2].analyze_current_regime(
                            timeframe
                        )

                        opportunity = {
                            "pair": f"{symbol1}/{symbol2}",
                            "correlation": correlation,
                            "spread_zscore": current_zscore,
                            "signal": (
                                "LONG_1_SHORT_2"
                                if current_zscore < -2
                                else "SHORT_1_LONG_2"
                            ),
                            "confidence_1": analysis1.regime_confidence,
                            "confidence_2": analysis2.regime_confidence,
                            "regime_1": analysis1.current_regime.value,
                            "regime_2": analysis2.current_regime.value,
                            "opportunity_strength": abs(current_zscore)
                            * min(
                                analysis1.regime_confidence, analysis2.regime_confidence
                            ),
                        }

                        opportunities.append(opportunity)

                    except Exception as e:
                        print(f"Error analyzing pair {symbol1}/{symbol2}: {e!s}")
                        continue

            except Exception as e:
                print(f"Error processing pair {symbol1}/{symbol2}: {e!s}")
                continue

        # Sort by opportunity strength
        opportunities.sort(key=lambda x: x["opportunity_strength"], reverse=True)

        return opportunities[:5]  # Return top 5 opportunities

    def print_portfolio_summary(self, timeframe: str = "1D") -> None:
        """
        Print comprehensive portfolio analysis.

        Args:
            timeframe: Timeframe for analysis
        """
        print("\n" + "=" * 100)
        print(f"PORTFOLIO HMM REGIME ANALYSIS ({timeframe})")
        print("=" * 100)

        # Portfolio overview
        print(f"Portfolio: {', '.join(self.symbols)}")
        print(f"Active Symbols: {len(self.analyzers)}")

        # Portfolio metrics
        summary = self.get_portfolio_regime_summary(timeframe)

        print("\n📊 PORTFOLIO REGIME SUMMARY:")
        print(f"   Dominant Regime: {summary['dominant_regime']}")
        print(f"   Regime Consensus: {summary['regime_consensus']:.1%}")
        print(f"   Average Confidence: {summary['average_confidence']:.1%}")
        print(f"   Portfolio Risk: {summary['risk_level']}")
        print(f"   Diversification Benefit: {summary['diversification_benefit']:.1%}")
        print(f"   Correlation Risk: {summary['correlation_risk']:.1%}")

        # Regime distribution
        if summary["regime_distribution"]:
            print("\n📈 REGIME DISTRIBUTION:")
            for regime, count in summary["regime_distribution"].items():
                percentage = count / len(self.analyzers) * 100
                print(f"   {regime}: {count} assets ({percentage:.1f}%)")

        # Individual symbol analysis
        print("\n🔍 INDIVIDUAL SYMBOL ANALYSIS:")
        for symbol, analyzer in self.analyzers.items():
            try:
                analysis = analyzer.analyze_current_regime(timeframe)
                price = analyzer.data[timeframe]["Close"].iloc[-1]
                print(
                    f"   {symbol}: ${price:.2f} | {analysis.current_regime.value} | "
                    f"Conf: {analysis.regime_confidence:.1%} | "
                    f"Strategy: {analysis.recommended_strategy.value}"
                )
            except Exception as e:
                print(f"   {symbol}: Error - {e!s}")

        # Statistical arbitrage opportunities
        arbitrage_pairs = self.identify_arbitrage_pairs(timeframe)
        if arbitrage_pairs:
            print("\n💰 STATISTICAL ARBITRAGE OPPORTUNITIES:")
            for i, opp in enumerate(arbitrage_pairs[:3], 1):
                print(
                    f"   {i}. {opp['pair']}: {opp['signal']} "
                    f"(Z-score: {opp['spread_zscore']:.2f}, "
                    f"Strength: {opp['opportunity_strength']:.3f})"
                )

        # Correlation analysis
        try:
            correlations = self.calculate_regime_correlations(timeframe)
            print("\n🔗 CORRELATION INSIGHTS:")

            # Find highest and lowest correlations
            if len(correlations) > 1:
                price_corr_cols = [
                    col for col in correlations.columns if "_price_corr" in col
                ]
                if price_corr_cols:
                    print("   High correlation pairs (potential risk concentration)")
                    print("   Low correlation pairs (diversification opportunities)")

        except Exception as e:
            print(f"   Correlation analysis error: {e!s}")

        print("=" * 100)
