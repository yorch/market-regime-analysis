"""
Portfolio HMM analyzer for multi-asset regime analysis.

This module implements portfolio-level analysis following Renaissance
Technologies' approach to multi-asset regime detection and correlation analysis.
"""

import logging
import warnings
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from mra_lib.analyzer import MarketRegimeAnalyzer
from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.enums import MarketRegime

logger = logging.getLogger(__name__)

# Derived (non-symbol) columns stored alongside prices in ``portfolio_data``.
_DERIVED_COLUMNS = frozenset({"portfolio_return", "portfolio_volatility"})

_MIN_PAIR_OBSERVATIONS = 50


class PortfolioHMMAnalyzer:
    """
    Multi-asset regime analysis following Renaissance approach.

    This class extends the single-asset analysis to portfolio-level
    regime detection, correlation analysis, and statistical arbitrage
    pair identification across multiple assets.
    """

    def __init__(
        self,
        symbols: list[str],
        periods: dict[str, str] | None = None,
        provider_flag: str = "yfinance",
        api_key: str | None = None,
    ) -> None:
        """
        Initialize portfolio analyzer.

        Args:
            symbols: List of trading symbols to analyze
            periods: Dictionary mapping timeframes to data periods
            provider_flag: 'yfinance' or 'alphavantage'
            api_key: API key for Alpha Vantage (if needed)
        """
        self.symbols = symbols
        self.periods = periods
        self.analyzers: dict[str, MarketRegimeAnalyzer] = {}
        self.portfolio_data: dict[str, pd.DataFrame] = {}
        # Most recent failure per symbol (initialization or analysis), so callers can
        # report why symbols are missing instead of only that they are.
        self.errors: dict[str, Exception] = {}

        print(f"Initializing portfolio analysis for {len(symbols)} symbols...")

        # Initialize individual analyzers
        for symbol in symbols:
            try:
                analyzer = MarketRegimeAnalyzer(
                    symbol, periods, provider_flag=provider_flag, api_key=api_key
                )
                self.analyzers[symbol] = analyzer
                print(f"✓ Initialized {symbol}")
            except Exception as e:
                self.errors[symbol] = e
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
                print(f"✓ Prepared {timeframe} portfolio data: {len(portfolio_df)} periods")

    def _symbol_columns(self, timeframe: str) -> list[str]:
        """Price columns (symbols) in ``portfolio_data[timeframe]``, excluding derived ones."""
        if timeframe not in self.portfolio_data:
            return []
        return [c for c in self.portfolio_data[timeframe].columns if c not in _DERIVED_COLUMNS]

    def collect_analyses(self, timeframe: str) -> dict[str, RegimeAnalysis]:
        """
        Run ``analyze_current_regime`` once per symbol.

        Symbols whose analysis fails are logged, recorded in :attr:`errors`, and
        omitted. Pass the result to the other report methods to avoid re-running
        each symbol's analysis.
        """
        analyses: dict[str, RegimeAnalysis] = {}
        for symbol, analyzer in self.analyzers.items():
            try:
                analyses[symbol] = analyzer.analyze_current_regime(timeframe)
                self.errors.pop(symbol, None)
            except Exception as e:
                self.errors[symbol] = e
                logger.warning("Error analyzing %s: %s", symbol, e)
        return analyses

    def get_return_correlation_matrix(
        self, timeframe: str = "1D", symbols: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Pearson correlation matrix of simple returns (``pct_change``).

        Correlating price *levels* of trending/random-walk series produces
        spurious correlations; returns are (approximately) stationary.

        Args:
            timeframe: Timeframe to use
            symbols: Symbols to include (default: all symbols with data)

        Returns:
            Square DataFrame of return correlations (empty if no data)
        """
        cols = self._symbol_columns(timeframe)
        if symbols is not None:
            cols = [c for c in symbols if c in cols]
        if not cols:
            return pd.DataFrame()
        returns = self.portfolio_data[timeframe][cols].pct_change().dropna(how="all")
        corr: pd.DataFrame = returns.corr()
        return corr

    def calculate_regime_correlations(
        self, timeframe: str = "1D", analyses: dict[str, RegimeAnalysis] | None = None
    ) -> pd.DataFrame:
        """
        Calculate cross-asset regime info joined with return correlations.

        Args:
            timeframe: Timeframe for correlation analysis
            analyses: Optional precomputed per-symbol analyses (avoids recomputation)

        Returns:
            DataFrame indexed by symbol with regime columns plus one
            ``<symbol>_price_corr`` column per symbol. Despite the historical
            ``_price_corr`` suffix (kept for API compatibility), these are
            correlations of returns, not of price levels.
        """
        if timeframe not in self.portfolio_data:
            raise ValueError(f"Portfolio data not available for {timeframe}")

        if analyses is None:
            analyses = self.collect_analyses(timeframe)

        regime_data = {
            symbol: {
                "regime": analysis.current_regime.value,
                "confidence": analysis.regime_confidence,
                "state": analysis.hmm_state,
                "persistence": analysis.regime_persistence,
            }
            for symbol, analysis in analyses.items()
        }
        regime_df = pd.DataFrame(regime_data).T

        corr = self.get_return_correlation_matrix(timeframe, list(regime_data))
        if len(corr.columns) > 1:
            regime_df = regime_df.join(corr.add_suffix("_price_corr"))

        return regime_df

    def get_portfolio_regime_summary(
        self, timeframe: str = "1D", analyses: dict[str, RegimeAnalysis] | None = None
    ) -> dict[str, Any]:
        """
        Get portfolio-level regime metrics.

        Args:
            timeframe: Timeframe for analysis
            analyses: Optional precomputed per-symbol analyses (avoids recomputation)

        Returns:
            Dictionary with portfolio metrics. ``correlation_risk`` is the mean
            absolute pairwise *return* correlation.
        """
        summary: dict[str, Any] = {
            "dominant_regime": None,
            "regime_consensus": 0.0,
            "average_confidence": 0.0,
            "risk_level": "Unknown",
            "diversification_benefit": 0.0,
            "regime_distribution": {},
            "correlation_risk": 0.0,
        }

        try:
            if analyses is None:
                analyses = self.collect_analyses(timeframe)

            if not analyses:
                return summary

            # Calculate regime distribution
            regime_counts: dict[str, int] = {}
            for analysis in analyses.values():
                key = analysis.current_regime.value
                regime_counts[key] = regime_counts.get(key, 0) + 1

            summary["regime_distribution"] = regime_counts

            total_assets = len(analyses)
            dominant_regime = max(regime_counts.items(), key=lambda x: x[1])
            summary["dominant_regime"] = dominant_regime[0]
            summary["regime_consensus"] = dominant_regime[1] / total_assets

            summary["average_confidence"] = float(
                np.mean([a.regime_confidence for a in analyses.values()])
            )

            # Assess portfolio risk level
            risky = sum(
                1
                for a in analyses.values()
                if a.current_regime in (MarketRegime.HIGH_VOLATILITY, MarketRegime.UNKNOWN)
            )
            if risky / total_assets > 0.5:
                summary["risk_level"] = "High"
            elif risky / total_assets > 0.3:
                summary["risk_level"] = "Medium"
            else:
                summary["risk_level"] = "Low"

            # Correlation risk: needs at least two analyzed symbols with price data
            symbols_in_data = [s for s in analyses if s in self._symbol_columns(timeframe)]
            if len(symbols_in_data) > 1:
                corr_matrix = self.get_return_correlation_matrix(timeframe, symbols_in_data)
                n = len(corr_matrix)
                if n > 1:
                    # Average absolute correlation (excluding diagonal)
                    total_corr = corr_matrix.abs().to_numpy().sum() - n
                    avg_corr = float(total_corr / (n * (n - 1)))
                    if not np.isnan(avg_corr):
                        summary["correlation_risk"] = avg_corr
                        summary["diversification_benefit"] = 1.0 - avg_corr

        except Exception as e:
            logger.warning("Error calculating portfolio summary: %s", e)

        return summary

    @staticmethod
    def _cointegration_spread(
        price1: pd.Series, price2: pd.Series
    ) -> tuple[pd.Series, float, float, float] | None:
        """
        Engle-Granger spread of two price series.

        Regresses ``log(price1)`` on ``log(price2)`` (OLS with intercept) over
        all bars *except the last* and tests that in-sample residual for a unit
        root via ``statsmodels`` ``coint``. The last bar is then scored
        out-of-sample so a large current deviation is not absorbed into the fit.

        Returns:
            ``(in_sample_residual, hedge_ratio, coint_pvalue, current_residual)``
            or None when the series are too short or contain non-positive prices.
        """
        pair = pd.concat([price1, price2], axis=1).dropna()
        if len(pair) < _MIN_PAIR_OBSERVATIONS or (pair <= 0).to_numpy().any():
            return None
        log1_all = np.log(pair.iloc[:, 0])
        log2_all = np.log(pair.iloc[:, 1])
        log1, log2 = log1_all.iloc[:-1], log2_all.iloc[:-1]
        if log2.std() == 0 or log1.std() == 0:
            return None

        hedge_ratio, intercept = np.polyfit(log2.to_numpy(), log1.to_numpy(), 1)
        residual = log1 - (intercept + hedge_ratio * log2)
        current = float(log1_all.iloc[-1] - (intercept + hedge_ratio * log2_all.iloc[-1]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # statsmodels collinearity/convergence noise
            _, pvalue, _ = coint(log1, log2)

        return residual, float(hedge_ratio), float(pvalue), current

    def identify_arbitrage_pairs(
        self,
        timeframe: str = "1D",
        analyses: dict[str, RegimeAnalysis] | None = None,
        max_pvalue: float = 0.05,
        entry_zscore: float = 2.0,
    ) -> list[dict[str, Any]]:
        """
        Statistical arbitrage pairs detection (Engle-Granger cointegration).

        For each pair the log-price hedge-ratio residual
        ``log(p1) - (a + b * log(p2))`` is computed; the pair qualifies when the
        Engle-Granger test rejects "no cointegration" at ``max_pvalue`` and the
        current residual z-score exceeds ``entry_zscore`` in absolute value.
        A positive z-score means symbol 1 is rich vs. symbol 2.

        Args:
            timeframe: Timeframe for analysis
            analyses: Optional precomputed per-symbol analyses (avoids recomputation)
            max_pvalue: Maximum cointegration p-value to accept a pair
            entry_zscore: Minimum |z| of the current spread to signal

        Returns:
            Up to 5 opportunities, strongest first
        """
        opportunities: list[dict[str, Any]] = []

        if timeframe not in self.portfolio_data:
            return opportunities

        price_columns = self._symbol_columns(timeframe)
        available_symbols = [s for s in self.symbols if s in self.analyzers and s in price_columns]
        if len(available_symbols) < 2:
            return opportunities

        cache: dict[str, RegimeAnalysis] = dict(analyses or {})
        prices = self.portfolio_data[timeframe]

        def _analysis(symbol: str) -> RegimeAnalysis:
            if symbol not in cache:
                cache[symbol] = self.analyzers[symbol].analyze_current_regime(timeframe)
            return cache[symbol]

        for symbol1, symbol2 in combinations(available_symbols, 2):
            try:
                spread = self._cointegration_spread(prices[symbol1], prices[symbol2])
                if spread is None:
                    continue
                residual, hedge_ratio, pvalue, current_resid = spread
                if pvalue > max_pvalue:
                    continue

                resid_std = residual.std()
                if not resid_std > 0:
                    continue
                current_zscore = float((current_resid - residual.mean()) / resid_std)
                if np.isnan(current_zscore) or abs(current_zscore) <= entry_zscore:
                    continue

                analysis1 = _analysis(symbol1)
                analysis2 = _analysis(symbol2)
                return_corr = float(prices[symbol1].pct_change().corr(prices[symbol2].pct_change()))

                opportunities.append(
                    {
                        "pair": f"{symbol1}/{symbol2}",
                        "correlation": return_corr,
                        "hedge_ratio": hedge_ratio,
                        "coint_pvalue": pvalue,
                        "spread_zscore": current_zscore,
                        "signal": "LONG_1_SHORT_2" if current_zscore < 0 else "SHORT_1_LONG_2",
                        "confidence_1": analysis1.regime_confidence,
                        "confidence_2": analysis2.regime_confidence,
                        "regime_1": analysis1.current_regime.value,
                        "regime_2": analysis2.current_regime.value,
                        "opportunity_strength": abs(current_zscore)
                        * min(analysis1.regime_confidence, analysis2.regime_confidence),
                    }
                )

            except Exception as e:
                logger.warning("Error processing pair %s/%s: %s", symbol1, symbol2, e)
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

        # Analyze each symbol once and reuse for every section of the report
        analyses = self.collect_analyses(timeframe)

        # Portfolio metrics
        summary = self.get_portfolio_regime_summary(timeframe, analyses=analyses)

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
            analyzed = len(analyses)  # denominator = successful analyses
            for regime, count in summary["regime_distribution"].items():
                percentage = count / analyzed * 100
                print(f"   {regime}: {count} assets ({percentage:.1f}%)")

        # Individual symbol analysis
        print("\n🔍 INDIVIDUAL SYMBOL ANALYSIS:")
        for symbol, analyzer in self.analyzers.items():
            try:
                if symbol not in analyses:
                    raise ValueError("analysis failed")
                analysis = analyses[symbol]
                price = analyzer.data[timeframe]["Close"].iloc[-1]
                print(
                    f"   {symbol}: ${price:.2f} | {analysis.current_regime.value} | "
                    f"Conf: {analysis.regime_confidence:.1%} | "
                    f"Strategy: {analysis.recommended_strategy.value}"
                )
            except Exception as e:
                print(f"   {symbol}: Error - {e!s}")

        # Statistical arbitrage opportunities
        arbitrage_pairs = self.identify_arbitrage_pairs(timeframe, analyses=analyses)
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
            correlations = self.calculate_regime_correlations(timeframe, analyses=analyses)
            print("\n🔗 CORRELATION INSIGHTS:")

            # Highest and lowest pairwise return correlations
            corr = self.get_return_correlation_matrix(timeframe, list(correlations.index))
            pairs = [
                (f"{a}/{b}", float(corr.loc[a, b]))
                for a, b in combinations(corr.columns, 2)
                if not pd.isna(corr.loc[a, b])
            ]
            if pairs:
                high = max(pairs, key=lambda x: x[1])
                low = min(pairs, key=lambda x: x[1])
                print(f"   Highest return correlation: {high[0]} ({high[1]:.2f})")
                print(f"   Lowest return correlation: {low[0]} ({low[1]:.2f})")

        except Exception as e:
            print(f"   Correlation analysis error: {e!s}")

        print("=" * 100)
