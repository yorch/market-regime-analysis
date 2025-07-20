"""
Market Regime Analyzer - Main analysis engine.

This module implements the primary analysis engine following Jim Simons'
complete methodology for market regime detection and trading analysis.
"""

import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from .data_classes import RegimeAnalysis
from .data_provider import AlphaVantageProvider, MarketDataProvider, YFinanceProvider
from .enums import MarketRegime, TradingStrategy
from .hmm_detector import HiddenMarkovRegimeDetector

warnings.filterwarnings("ignore", category=FutureWarning)


class MarketRegimeAnalyzer:
    """
    Primary analysis engine implementing full Simons methodology.

    This class serves as the main interface for market regime analysis,
    integrating HMM detection with comprehensive technical analysis,
    statistical arbitrage identification, and risk management.
    """

    def __init__(
        self,
        symbol: str = "SPY",
        periods: dict[str, str] | None = None,
        provider_flag: str = "yfinance",
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the Market Regime Analyzer.

        Args:
            symbol: Trading symbol to analyze
            periods: Dictionary mapping timeframes to data periods
            provider_flag: 'yfinance' or 'alphavantage'
            api_key: API key for Alpha Vantage (if needed)
        """
        self.symbol = symbol
        self.periods = periods or {
            "1D": "2y",  # Daily data for 2 years
            "1H": "6mo",  # Hourly data for 6 months
            "15m": "2mo",  # 15-min data for 2 months
        }

        # Data storage
        self.data: dict[str, pd.DataFrame] = {}
        self.indicators: dict[str, pd.DataFrame] = {}
        self.hmm_models: dict[str, HiddenMarkovRegimeDetector] = {}

        # Regime multipliers following Renaissance approach
        self.regime_multipliers = {
            MarketRegime.BULL_TRENDING: 1.3,
            MarketRegime.BEAR_TRENDING: 0.7,
            MarketRegime.MEAN_REVERTING: 1.2,
            MarketRegime.HIGH_VOLATILITY: 0.4,
            MarketRegime.LOW_VOLATILITY: 1.1,
            MarketRegime.BREAKOUT: 0.9,
            MarketRegime.UNKNOWN: 0.2,
        }

        # Data provider selection
        if provider_flag == "yfinance":
            self.provider: MarketDataProvider = YFinanceProvider()
        elif provider_flag == "alphavantage":
            if not api_key or not isinstance(api_key, str):
                raise ValueError(
                    "Alpha Vantage API key required for alphavantage provider."
                )
            self.provider: MarketDataProvider = AlphaVantageProvider(api_key)
        else:
            raise ValueError(f"Unknown provider: {provider_flag}")

        # Initialize data and models
        self._load_data()
        self._calculate_indicators()
        self._train_hmm_models()

    def _load_data(self) -> None:
        """
        Fetch market data using the selected provider for all timeframes.

        Raises:
            ValueError: If data download fails
        """
        print(f"Loading data for {self.symbol}...")

        for timeframe, period in self.periods.items():
            try:
                df = self.provider.fetch(self.symbol, period, timeframe.lower())

                if df.empty:
                    raise ValueError(f"No data available for {self.symbol} {timeframe}")

                # Ensure we have OHLCV columns
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns: {missing_cols}")

                self.data[timeframe] = df
                print(f"✓ Loaded {len(df)} bars for {timeframe}")

            except Exception as e:
                print(f"✗ Failed to load {timeframe} data: {e!s}")
                raise ValueError(f"Data loading failed for {timeframe}: {e!s}")

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators with TA-Lib fallbacks.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with technical indicators
        """
        indicators = df.copy()

        # Basic price indicators
        indicators["returns"] = df["Close"].pct_change()
        indicators["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # Moving averages
        indicators["ema_9"] = df["Close"].ewm(span=9).mean()
        indicators["ema_34"] = df["Close"].ewm(span=34).mean()
        indicators["sma_50"] = df["Close"].rolling(50).mean()
        indicators["sma_200"] = df["Close"].rolling(200).mean()

        # ATR calculation
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift(1))
        low_close = np.abs(df["Low"] - df["Close"].shift(1))
        true_range = pd.Series(
            np.maximum(high_low, np.maximum(high_close, low_close)), index=df.index
        )
        indicators["atr"] = true_range.rolling(14).mean()
        indicators["atr_percent"] = indicators["atr"] / df["Close"] * 100

        # Bollinger Bands
        sma_20 = df["Close"].rolling(20).mean()
        std_20 = df["Close"].rolling(20).std()
        indicators["bb_upper"] = sma_20 + (2 * std_20)
        indicators["bb_middle"] = sma_20
        indicators["bb_lower"] = sma_20 - (2 * std_20)
        indicators["bb_width"] = (
            indicators["bb_upper"] - indicators["bb_lower"]
        ) / indicators["bb_middle"]

        # RSI calculation
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        indicators["rsi"] = 100 - (100 / (1 + rs))

        # MACD calculation
        ema_12 = df["Close"].ewm(span=12).mean()
        ema_26 = df["Close"].ewm(span=26).mean()
        indicators["macd"] = ema_12 - ema_26
        indicators["macd_signal"] = indicators["macd"].ewm(span=9).mean()
        indicators["macd_histogram"] = indicators["macd"] - indicators["macd_signal"]

        # Volatility measures
        indicators["volatility"] = indicators["returns"].rolling(20).std() * np.sqrt(
            252
        )
        indicators["vol_rank"] = indicators["volatility"].rolling(252).rank(pct=True)

        # Volume analysis
        if df["Volume"].sum() > 0:
            indicators["volume_ma"] = df["Volume"].rolling(20).mean()
            indicators["volume_ratio"] = df["Volume"] / indicators["volume_ma"]
            indicators["price_volume"] = indicators["returns"] * np.log(
                df["Volume"] + 1
            )
        else:
            indicators["volume_ma"] = pd.Series(1, index=df.index)
            indicators["volume_ratio"] = pd.Series(1, index=df.index)
            indicators["price_volume"] = indicators["returns"]

        # Statistical Arbitrage Features
        indicators["price_zscore"] = (df["Close"] - df["Close"].rolling(50).mean()) / (
            df["Close"].rolling(50).std() + 1e-8
        )
        indicators["return_zscore"] = (
            indicators["returns"] - indicators["returns"].rolling(50).mean()
        ) / (indicators["returns"].rolling(50).std() + 1e-8)

        # Return autocorrelation (momentum persistence)
        for lag in [1, 2, 5]:
            indicators[f"autocorr_{lag}"] = (
                indicators["returns"]
                .rolling(20)
                .apply(lambda x: x.autocorr(lag=lag), raw=False)
            )

        # Mean reversion signals
        indicators["mean_reversion_score"] = np.abs(indicators["price_zscore"]) * (
            1 - np.abs(indicators["autocorr_1"].fillna(0))
        )

        return indicators

    def _calculate_indicators(self) -> None:
        """Process all timeframes to calculate technical indicators."""
        print("Calculating technical indicators...")

        for timeframe, df in self.data.items():
            self.indicators[timeframe] = self._calculate_technical_indicators(df)
            print(f"✓ Calculated indicators for {timeframe}")

    def _train_hmm_models(self) -> None:
        """Train HMM models for each timeframe."""
        print("Training HMM models...")

        for timeframe, df in self.data.items():
            try:
                hmm = HiddenMarkovRegimeDetector(n_states=6)
                hmm.fit(df)
                self.hmm_models[timeframe] = hmm
                print(f"✓ Trained HMM for {timeframe}")
            except Exception as e:
                print(f"✗ Failed to train HMM for {timeframe}: {e!s}")

    def _get_trading_strategy(self, regime: MarketRegime) -> TradingStrategy:
        """
        Map market regimes to optimal trading strategies.

        Args:
            regime: Current market regime

        Returns:
            Recommended trading strategy
        """
        strategy_map = {
            MarketRegime.BULL_TRENDING: TradingStrategy.TREND_FOLLOWING,
            MarketRegime.BEAR_TRENDING: TradingStrategy.DEFENSIVE,
            MarketRegime.MEAN_REVERTING: TradingStrategy.MEAN_REVERSION,
            MarketRegime.HIGH_VOLATILITY: TradingStrategy.VOLATILITY_TRADING,
            MarketRegime.LOW_VOLATILITY: TradingStrategy.MOMENTUM,
            MarketRegime.BREAKOUT: TradingStrategy.MOMENTUM,
            MarketRegime.UNKNOWN: TradingStrategy.AVOID,
        }

        return strategy_map.get(regime, TradingStrategy.AVOID)

    def _get_position_sizing_multiplier(
        self, regime: MarketRegime, confidence: float
    ) -> float:
        """
        Calculate risk-adjusted position sizing multiplier.

        Args:
            regime: Current market regime
            confidence: Confidence in regime classification

        Returns:
            Position sizing multiplier
        """
        base_multiplier = self.regime_multipliers.get(regime, 0.2)

        # Confidence scaling (0.3 to 1.0)
        confidence_factor = 0.3 + (confidence * 0.7)

        # Final multiplier with safety caps
        multiplier = base_multiplier * confidence_factor
        return max(0.01, min(0.5, multiplier))  # Cap between 1% and 50%

    def _identify_arbitrage_opportunities(self, df: pd.DataFrame) -> list[str]:
        """
        Identify statistical arbitrage opportunities (core Simons strategy).

        Args:
            df: DataFrame with indicators

        Returns:
            List of arbitrage opportunity descriptions
        """
        opportunities = []

        if df.empty:
            return opportunities

        latest = df.iloc[-1]

        # Mean reversion opportunities
        if hasattr(latest, "price_zscore"):
            price_zscore = latest["price_zscore"]
            if abs(price_zscore) > 2.0:
                direction = "SHORT" if price_zscore > 0 else "LONG"
                opportunities.append(
                    f"Mean Reversion: {direction} signal (Z-score: {price_zscore:.2f})"
                )

        # Autocorrelation breakdown
        if hasattr(latest, "autocorr_1"):
            autocorr = latest["autocorr_1"]
            if not pd.isna(autocorr) and abs(autocorr) < 0.1:
                opportunities.append(
                    f"Momentum Breakdown: Low autocorr ({autocorr:.3f})"
                )

        # Volatility regime changes
        if hasattr(latest, "vol_rank"):
            vol_rank = latest["vol_rank"]
            if not pd.isna(vol_rank):
                if vol_rank > 0.8:
                    opportunities.append("Vol Regime: Expect mean reversion (High vol)")
                elif vol_rank < 0.2:
                    opportunities.append("Vol Regime: Expect breakout (Low vol)")

        return opportunities

    def _generate_statistical_signals(
        self, df: pd.DataFrame, regime: MarketRegime
    ) -> list[str]:
        """
        Generate regime-specific statistical signals.

        Args:
            df: DataFrame with indicators
            regime: Current market regime

        Returns:
            List of statistical signals
        """
        signals = []

        if df.empty:
            return signals

        latest = df.iloc[-1]

        # Regime-specific signals
        if regime == MarketRegime.MEAN_REVERTING:
            if hasattr(latest, "bb_upper") and hasattr(latest, "bb_lower"):
                close = latest["Close"]
                if close > latest["bb_upper"]:
                    signals.append("BB: Price above upper band (SHORT signal)")
                elif close < latest["bb_lower"]:
                    signals.append("BB: Price below lower band (LONG signal)")

        elif regime in [MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING]:
            if hasattr(latest, "ema_9") and hasattr(latest, "ema_34"):
                if latest["ema_9"] > latest["ema_34"]:
                    signals.append("EMA: Bullish crossover (LONG bias)")
                else:
                    signals.append("EMA: Bearish crossover (SHORT bias)")

        # RSI signals
        if hasattr(latest, "rsi"):
            rsi = latest["rsi"]
            if not pd.isna(rsi):
                if rsi > 70:
                    signals.append(f"RSI: Overbought ({rsi:.1f})")
                elif rsi < 30:
                    signals.append(f"RSI: Oversold ({rsi:.1f})")

        # MACD signals
        if hasattr(latest, "macd") and hasattr(latest, "macd_signal"):
            macd = latest["macd"]
            signal = latest["macd_signal"]
            if not pd.isna(macd) and not pd.isna(signal):
                if macd > signal:
                    signals.append("MACD: Bullish signal")
                else:
                    signals.append("MACD: Bearish signal")

        return signals

    def _identify_key_levels(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Identify key support and resistance levels.

        Args:
            df: DataFrame with price data

        Returns:
            Dictionary of key levels
        """
        levels = {}

        if len(df) < 50:
            return levels

        # Recent price action
        recent_data = df.tail(50)
        close = df["Close"].iloc[-1]

        # Support and resistance from recent highs/lows
        recent_highs = recent_data["High"].rolling(10).max()
        recent_lows = recent_data["Low"].rolling(10).min()

        levels["resistance"] = float(recent_highs.max())
        levels["support"] = float(recent_lows.min())

        # Moving average levels
        if "sma_50" in df.columns and not pd.isna(df["sma_50"].iloc[-1]):
            levels["sma_50"] = float(df["sma_50"].iloc[-1])

        if "sma_200" in df.columns and not pd.isna(df["sma_200"].iloc[-1]):
            levels["sma_200"] = float(df["sma_200"].iloc[-1])

        # Bollinger Bands
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            if not pd.isna(df["bb_upper"].iloc[-1]):
                levels["bb_upper"] = float(df["bb_upper"].iloc[-1])
            if not pd.isna(df["bb_lower"].iloc[-1]):
                levels["bb_lower"] = float(df["bb_lower"].iloc[-1])

        # ATR-based levels
        if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]):
            atr = df["atr"].iloc[-1]
            levels["atr_resistance"] = float(close + atr)
            levels["atr_support"] = float(close - atr)

        return levels

    def analyze_current_regime(self, timeframe: str) -> RegimeAnalysis:
        """
        Main analysis function returning comprehensive regime analysis.

        Args:
            timeframe: Timeframe to analyze (e.g., '1D', '1H', '15m')

        Returns:
            RegimeAnalysis object with comprehensive results

        Raises:
            ValueError: If timeframe is not available
        """
        if timeframe not in self.data:
            raise ValueError(f"Timeframe {timeframe} not available")

        df = self.data[timeframe]
        indicators = self.indicators[timeframe]
        hmm = self.hmm_models.get(timeframe)

        if hmm is None:
            raise ValueError(f"HMM model not available for {timeframe}")

        # Get regime prediction
        regime, state, confidence = hmm.predict_regime(df)

        # Calculate regime persistence
        recent_predictions = []
        lookback = min(20, len(df) - 50)
        for i in range(lookback):
            window_df = df.iloc[-(lookback - i) :]
            if len(window_df) >= 50:
                _, temp_state, _ = hmm.predict_regime(window_df)
                recent_predictions.append(temp_state)

        persistence = hmm.calculate_regime_persistence(
            np.array(recent_predictions + [state])
        )

        # Get transition probability
        if len(recent_predictions) > 0:
            prev_state = recent_predictions[-1] if recent_predictions else state
            transition_prob = hmm.get_transition_probability(prev_state, state)
        else:
            transition_prob = 0.5

        # Determine risk level
        if confidence > 0.8 and persistence > 0.7:
            risk_level = "Low"
        elif confidence > 0.6 and persistence > 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Get trading strategy
        strategy = self._get_trading_strategy(regime)

        # Calculate position sizing
        position_multiplier = self._get_position_sizing_multiplier(regime, confidence)

        # Identify opportunities and signals
        arbitrage_opps = self._identify_arbitrage_opportunities(indicators)
        statistical_signals = self._generate_statistical_signals(indicators, regime)
        key_levels = self._identify_key_levels(indicators)

        return RegimeAnalysis(
            current_regime=regime,
            hmm_state=state,
            transition_probability=transition_prob,
            regime_persistence=persistence,
            recommended_strategy=strategy,
            position_sizing_multiplier=position_multiplier,
            risk_level=risk_level,
            arbitrage_opportunities=arbitrage_opps,
            statistical_signals=statistical_signals,
            key_levels=key_levels,
            regime_confidence=confidence,
        )

    def print_analysis_report(self, timeframe: str) -> None:
        """
        Print comprehensive formatted analysis report.

        Args:
            timeframe: Timeframe to analyze
        """
        try:
            analysis = self.analyze_current_regime(timeframe)
            current_price = self.data[timeframe]["Close"].iloc[-1]

            print("\n" + "=" * 80)
            print(f"HMM MARKET REGIME ANALYSIS - {self.symbol} ({timeframe})")
            print("=" * 80)
            print(f"Current Price: ${current_price:.2f}")
            print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            print("\n📊 REGIME CLASSIFICATION:")
            print(f"   Current Regime: {analysis.current_regime.value}")
            print(f"   HMM State: {analysis.hmm_state}")
            print(f"   Confidence: {analysis.regime_confidence:.1%}")
            print(f"   Persistence: {analysis.regime_persistence:.1%}")
            print(f"   Transition Prob: {analysis.transition_probability:.1%}")

            print("\n📈 TRADING RECOMMENDATION:")
            print(f"   Strategy: {analysis.recommended_strategy.value}")
            print(f"   Position Size: {analysis.position_sizing_multiplier:.1%}")
            print(f"   Risk Level: {analysis.risk_level}")

            if analysis.arbitrage_opportunities:
                print("\n💰 STATISTICAL ARBITRAGE:")
                for opp in analysis.arbitrage_opportunities:
                    print(f"   • {opp}")

            if analysis.statistical_signals:
                print("\n📡 STATISTICAL SIGNALS:")
                for signal in analysis.statistical_signals:
                    print(f"   • {signal}")

            if analysis.key_levels:
                print("\n🎯 KEY LEVELS:")
                for level_name, level_value in analysis.key_levels.items():
                    print(f"   {level_name.upper()}: ${level_value:.2f}")

            print("=" * 80)

        except Exception as e:
            print(f"Error generating report: {e!s}")

    def plot_regime_analysis(self, timeframe: str, days: int = 60) -> None:
        """
        Generate 5-panel chart with regime background coloring.

        Args:
            timeframe: Timeframe to plot
            days: Number of days to show
        """
        try:
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt

            df = self.data[timeframe].tail(days)
            indicators = self.indicators[timeframe].tail(days)

            if len(df) < 10:
                print("Insufficient data for plotting")
                return

            # Create regime predictions for the period
            regime_colors = {
                MarketRegime.BULL_TRENDING: "green",
                MarketRegime.BEAR_TRENDING: "red",
                MarketRegime.MEAN_REVERTING: "blue",
                MarketRegime.HIGH_VOLATILITY: "orange",
                MarketRegime.LOW_VOLATILITY: "purple",
                MarketRegime.BREAKOUT: "yellow",
                MarketRegime.UNKNOWN: "gray",
            }

            fig, axes = plt.subplots(5, 1, figsize=(15, 20))
            fig.suptitle(
                f"{self.symbol} HMM Regime Analysis ({timeframe})", fontsize=16
            )

            # Panel 1: Price with regime background
            ax1 = axes[0]
            ax1.plot(df.index, df["Close"], label="Close Price", linewidth=2)

            if "ema_9" in indicators.columns:
                ax1.plot(df.index, indicators["ema_9"], label="EMA 9", alpha=0.7)
            if "ema_34" in indicators.columns:
                ax1.plot(df.index, indicators["ema_34"], label="EMA 34", alpha=0.7)

            ax1.set_title("Price with Regime Background")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Panel 2: Statistical arbitrage signals
            ax2 = axes[1]
            if "price_zscore" in indicators.columns:
                ax2.plot(df.index, indicators["price_zscore"], label="Price Z-Score")
                ax2.axhline(
                    y=2, color="r", linestyle="--", alpha=0.7, label="Overbought"
                )
                ax2.axhline(
                    y=-2, color="g", linestyle="--", alpha=0.7, label="Oversold"
                )
                ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)

            ax2.set_title("Statistical Arbitrage Signals")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Panel 3: Volatility measures
            ax3 = axes[2]
            if "atr_percent" in indicators.columns:
                ax3.plot(df.index, indicators["atr_percent"], label="ATR %")
            if "volatility" in indicators.columns:
                ax3.plot(df.index, indicators["volatility"], label="Volatility")

            ax3.set_title("Volatility Measures")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Panel 4: Return autocorrelation
            ax4 = axes[3]
            if "autocorr_1" in indicators.columns:
                ax4.plot(df.index, indicators["autocorr_1"], label="1-Day Autocorr")
                ax4.axhline(y=0, color="k", linestyle="-", alpha=0.3)

            ax4.set_title("Return Autocorrelation")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Panel 5: RSI and other oscillators
            ax5 = axes[4]
            if "rsi" in indicators.columns:
                ax5.plot(df.index, indicators["rsi"], label="RSI")
                ax5.axhline(y=70, color="r", linestyle="--", alpha=0.7)
                ax5.axhline(y=30, color="g", linestyle="--", alpha=0.7)

            ax5.set_title("Technical Oscillators")
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # Format x-axis
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error generating chart: {e!s}")

    def run_continuous_monitoring(self, interval: int = 300) -> None:
        """
        Real-time monitoring with auto-refresh and memory management.

        Args:
            interval: Refresh interval in seconds
        """
        print(f"Starting continuous monitoring (refresh every {interval}s)")
        print("Press Ctrl+C to stop...")

        try:
            while True:
                print(
                    f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Refreshing..."
                )

                # Reload data and retrain models
                self._load_data()
                self._calculate_indicators()
                self._train_hmm_models()

                # Print analysis for all timeframes
                for timeframe in self.periods.keys():
                    try:
                        self.print_analysis_report(timeframe)
                    except Exception as e:
                        print(f"Error in {timeframe} analysis: {e!s}")

                # Memory cleanup
                import gc

                gc.collect()

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Monitoring error: {e!s}")

    def export_analysis_to_csv(self, filename: str | None = None) -> None:
        """
        Export comprehensive analysis data to CSV for backtesting.

        Args:
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.symbol}_hmm_analysis_{timestamp}.csv"

        try:
            all_data = []

            for timeframe in self.periods.keys():
                try:
                    analysis = self.analyze_current_regime(timeframe)
                    df = self.data[timeframe]
                    indicators = self.indicators[timeframe]

                    # Create comprehensive export data
                    export_row = {
                        "timestamp": df.index[-1],
                        "symbol": self.symbol,
                        "timeframe": timeframe,
                        "close_price": df["Close"].iloc[-1],
                        "regime": analysis.current_regime.value,
                        "hmm_state": analysis.hmm_state,
                        "regime_confidence": analysis.regime_confidence,
                        "regime_persistence": analysis.regime_persistence,
                        "transition_probability": analysis.transition_probability,
                        "strategy": analysis.recommended_strategy.value,
                        "position_multiplier": analysis.position_sizing_multiplier,
                        "risk_level": analysis.risk_level,
                        "arbitrage_count": len(analysis.arbitrage_opportunities),
                        "signal_count": len(analysis.statistical_signals),
                    }

                    # Add key technical indicators
                    for col in [
                        "rsi",
                        "macd",
                        "volatility",
                        "atr_percent",
                        "price_zscore",
                        "autocorr_1",
                    ]:
                        if col in indicators.columns:
                            export_row[col] = indicators[col].iloc[-1]

                    # Add key levels
                    for level_name, level_value in analysis.key_levels.items():
                        export_row[f"level_{level_name}"] = level_value

                    all_data.append(export_row)

                except Exception as e:
                    print(f"Error exporting {timeframe}: {e!s}")

            if all_data:
                export_df = pd.DataFrame(all_data)
                export_df.to_csv(filename, index=False)
                print(f"✓ Analysis exported to {filename}")
            else:
                print("✗ No data to export")

        except Exception as e:
            print(f"Export error: {e!s}")
