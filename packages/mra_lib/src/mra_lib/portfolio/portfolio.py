"""
Portfolio HMM analyzer for multi-asset regime analysis.

This module implements portfolio-level multi-asset regime detection and
return-correlation / cointegration analysis.
"""

import logging
import warnings
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from mra_lib._deprecation import write_deprecated_report
from mra_lib.analyzer import MarketRegimeAnalyzer
from mra_lib.config.data_classes import RegimeAnalysis
from mra_lib.config.enums import MarketRegime
from mra_lib.errors import AuthError, DataLoadError, InvalidSymbolError, RateLimitError

logger = logging.getLogger(__name__)

# Derived (non-symbol) columns stored alongside prices in ``portfolio_data``.
_DERIVED_COLUMNS = frozenset({"portfolio_return", "portfolio_volatility"})

_MIN_PAIR_OBSERVATIONS = 50


class PortfolioHMMAnalyzer:
    """
    Multi-asset regime analysis.

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

        Symbols that fail to load are logged (WARNING), skipped, and recorded in
        :attr:`failed_symbols` (symbol -> exception); check :attr:`analyzers`
        to see which symbols are usable.

        Raises:
            Exception: If *every* symbol fails to load, the most actionable
                failure is re-raised unchanged so callers can map it by type:
                any ``AuthError``, else any ``RateLimitError``, else a
                ``ConnectionError``/``TimeoutError`` if any failure was transient,
                else ``InvalidSymbolError`` if all symbols were invalid. Otherwise a
                ``DataLoadError`` listing each symbol's failure is raised, chained
                to the first.
        """
        self.symbols = symbols
        self.periods = periods
        self.analyzers: dict[str, MarketRegimeAnalyzer] = {}
        self.failed_symbols: dict[str, Exception] = {}
        #: Symbols whose ``analyze_current_regime`` failed in :meth:`collect_analyses`
        self.analysis_failures: dict[str, Exception] = {}
        #: Most recent failure per symbol (initialization or analysis), so callers can
        #: report why symbols are missing instead of only that they are.
        self.errors: dict[str, Exception] = {}
        self.portfolio_data: dict[str, pd.DataFrame] = {}

        logger.info("Initializing portfolio analysis for %d symbols...", len(symbols))

        # Initialize individual analyzers
        for symbol in symbols:
            try:
                analyzer = MarketRegimeAnalyzer(
                    symbol, periods, provider_flag=provider_flag, api_key=api_key
                )
            except Exception as e:  # noqa: BLE001 - one bad symbol must not sink the portfolio
                self.failed_symbols[symbol] = e
                self.errors[symbol] = e
                logger.warning("✗ Failed to initialize %s: %s", symbol, e)
                continue
            self.analyzers[symbol] = analyzer
            logger.info("✓ Initialized %s", symbol)

        if symbols and not self.analyzers:
            self._raise_all_failed()

        self._prepare_portfolio_data()

    def _raise_all_failed(self) -> None:
        """Raise the root cause when no symbol could be loaded (see ``__init__``)."""
        failures = list(self.failed_symbols.values())
        # Most actionable cause first: credentials, then throttling, then outages,
        # then bad symbols; a mix of anything else is a DataLoadError
        for kind in (AuthError, RateLimitError):
            for error in failures:
                if isinstance(error, kind):
                    raise error
        for kinds in ((ConnectionError, TimeoutError), (InvalidSymbolError,)):
            if all(isinstance(e, kinds) for e in failures):
                raise failures[0]
        if any(isinstance(e, ConnectionError | TimeoutError) for e in failures):
            # Partly transient (e.g. outage + bad symbol): retrying may help
            raise next(e for e in failures if isinstance(e, ConnectionError | TimeoutError))
        first = failures[0]
        detail = "; ".join(f"{sym}: {e}" for sym, e in self.failed_symbols.items())
        raise DataLoadError(f"No symbol could be loaded ({detail})") from first

    def _prepare_portfolio_data(self) -> None:
        """Prepare aligned portfolio data for correlation analysis."""
        logger.info("Preparing portfolio correlation data...")

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
                logger.info(
                    "✓ Prepared %s portfolio data: %d periods", timeframe, len(portfolio_df)
                )

    def _symbol_columns(self, timeframe: str) -> list[str]:
        """Price columns (symbols) in ``portfolio_data[timeframe]``, excluding derived ones."""
        if timeframe not in self.portfolio_data:
            return []
        return [c for c in self.portfolio_data[timeframe].columns if c not in _DERIVED_COLUMNS]

    def collect_analyses(self, timeframe: str) -> dict[str, RegimeAnalysis]:
        """
        Run ``analyze_current_regime`` once per symbol.

        Symbols whose analysis fails are logged, omitted, and recorded in
        :attr:`analysis_failures` and :attr:`errors`. Pass the result to the other report methods
        to avoid re-running each symbol's analysis.
        """
        analyses: dict[str, RegimeAnalysis] = {}
        self.analysis_failures = {}
        for symbol, analyzer in self.analyzers.items():
            try:
                analyses[symbol] = analyzer.analyze_current_regime(timeframe)
                self.errors.pop(symbol, None)
            except Exception as e:  # noqa: BLE001 - documented: failed symbols are omitted
                self.analysis_failures[symbol] = e
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

        if analyses is not None:
            # Symbols that failed in collect_analyses are skipped, not re-run
            available_symbols = [s for s in available_symbols if s in analyses]

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

            except (ValueError, ArithmeticError) as e:
                # Cointegration/analysis failure for one pair (statsmodels, LinAlgError,
                # unfitted model): skip the pair, keep scanning the others
                logger.warning("Error processing pair %s/%s: %s", symbol1, symbol2, e)
                continue

        # Sort by opportunity strength
        opportunities.sort(key=lambda x: x["opportunity_strength"], reverse=True)

        return opportunities[:5]  # Return top 5 opportunities

    def format_portfolio_summary(self, timeframe: str = "1D") -> str:
        """
        Format a human-readable portfolio analysis report.

        Args:
            timeframe: Timeframe for analysis

        Returns:
            Multi-line report text (starts with a blank line, no trailing newline)
        """
        lines = [
            "",
            "=" * 100,
            f"PORTFOLIO HMM REGIME ANALYSIS ({timeframe})",
            "=" * 100,
            # Portfolio overview
            f"Portfolio: {', '.join(self.symbols)}",
            f"Active Symbols: {len(self.analyzers)}",
        ]

        # Analyze each symbol once and reuse for every section of the report
        analyses = self.collect_analyses(timeframe)

        # Portfolio metrics
        summary = self.get_portfolio_regime_summary(timeframe, analyses=analyses)

        lines += [
            "",
            "📊 PORTFOLIO REGIME SUMMARY:",
            f"   Dominant Regime: {summary['dominant_regime']}",
            f"   Regime Consensus: {summary['regime_consensus']:.1%}",
            f"   Average Confidence: {summary['average_confidence']:.1%}",
            f"   Portfolio Risk: {summary['risk_level']}",
            f"   Diversification Benefit: {summary['diversification_benefit']:.1%}",
            f"   Correlation Risk: {summary['correlation_risk']:.1%}",
        ]

        # Regime distribution
        if summary["regime_distribution"]:
            lines += ["", "📈 REGIME DISTRIBUTION:"]
            analyzed = len(analyses)  # denominator = successful analyses
            for regime, count in summary["regime_distribution"].items():
                percentage = count / analyzed * 100
                lines.append(f"   {regime}: {count} assets ({percentage:.1f}%)")

        # Individual symbol analysis
        lines += ["", "🔍 INDIVIDUAL SYMBOL ANALYSIS:"]
        for symbol, analyzer in self.analyzers.items():
            if symbol not in analyses:
                lines.append(f"   {symbol}: Error - analysis failed")
                continue
            if timeframe not in analyzer.data or analyzer.data[timeframe].empty:
                lines.append(f"   {symbol}: Error - no {timeframe} data")
                continue
            analysis = analyses[symbol]
            price = analyzer.data[timeframe]["Close"].iloc[-1]
            lines.append(
                f"   {symbol}: ${price:.2f} | {analysis.current_regime.value} | "
                f"Conf: {analysis.regime_confidence:.1%} | "
                f"Strategy: {analysis.recommended_strategy.value}"
            )

        # Statistical arbitrage opportunities
        arbitrage_pairs = self.identify_arbitrage_pairs(timeframe, analyses=analyses)
        if arbitrage_pairs:
            lines += ["", "💰 STATISTICAL ARBITRAGE OPPORTUNITIES:"]
            for i, opp in enumerate(arbitrage_pairs[:3], 1):
                lines.append(
                    f"   {i}. {opp['pair']}: {opp['signal']} "
                    f"(Z-score: {opp['spread_zscore']:.2f}, "
                    f"Strength: {opp['opportunity_strength']:.3f})"
                )

        # Correlation analysis
        try:
            correlations = self.calculate_regime_correlations(timeframe, analyses=analyses)
        except ValueError as e:  # no portfolio data for this timeframe
            lines.append(f"   Correlation analysis error: {e!s}")
        else:
            lines += ["", "🔗 CORRELATION INSIGHTS:"]

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
                lines.append(f"   Highest return correlation: {high[0]} ({high[1]:.2f})")
                lines.append(f"   Lowest return correlation: {low[0]} ({low[1]:.2f})")

        lines.append("=" * 100)
        return "\n".join(lines)

    def print_portfolio_summary(self, timeframe: str = "1D") -> None:
        """
        Print the portfolio report to stdout.

        .. deprecated::
            The library no longer prints. Use :meth:`format_portfolio_summary` and
            print or log the returned string.
        """
        write_deprecated_report(
            self.format_portfolio_summary(timeframe),
            "PortfolioHMMAnalyzer.print_portfolio_summary",
            "format_portfolio_summary",
        )
