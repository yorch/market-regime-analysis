#!/usr/bin/env python3
"""
Market Regime Analysis System
A comprehensive tool for analyzing market regimes and providing trading suggestions
Based on Jim Simons' Hidden Markov Model approach and modern regime detection techniques
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import os
from dataclasses import dataclass
from enum import Enum

# Scientific computing
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import seaborn as sns

# Technical analysis
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using custom implementations.")


class MarketRegime(Enum):
    """Market regime classifications"""

    BULL_QUIET = "Bull Quiet"
    BULL_VOLATILE = "Bull Volatile"
    BEAR_QUIET = "Bear Quiet"
    BEAR_VOLATILE = "Bear Volatile"
    SIDEWAYS_QUIET = "Sideways Quiet"
    SIDEWAYS_VOLATILE = "Sideways Volatile"
    UNKNOWN = "Unknown"


class TradingStrategy(Enum):
    """Trading strategy recommendations"""

    TREND_FOLLOWING = "Trend Following"
    MOMENTUM = "Momentum"
    MEAN_REVERSION = "Mean Reversion"
    BREAKOUT = "Breakout"
    DEFENSIVE = "Defensive"
    AVOID = "Avoid Trading"


@dataclass
class RegimeAnalysis:
    """Container for regime analysis results"""

    current_regime: MarketRegime
    confidence: float
    trend_direction: str
    volatility_level: str
    recommended_strategy: TradingStrategy
    position_sizing_multiplier: float
    risk_level: str
    entry_signals: List[str]
    exit_signals: List[str]
    key_levels: Dict[str, float]


class MarketRegimeAnalyzer:
    """
    Advanced Market Regime Analyzer based on Jim Simons' methodology
    Implements Hidden Markov Models and multi-timeframe analysis
    """

    def __init__(self, symbol: str = "SPY", periods: Dict[str, str] = None):
        self.symbol = symbol
        self.periods = periods or {
            "1D": "1y",  # Daily data for 1 year
            "1H": "3mo",  # Hourly data for 3 months
            "15m": "1mo",  # 15-min data for 1 month
            "5m": "5d",  # 5-min data for 5 days
        }

        self.data = {}
        self.regime_history = {}
        self.indicators = {}

        # Regime detection parameters
        self.ma_fast = 9
        self.ma_slow = 34
        self.atr_period = 14
        self.volatility_lookback = 20

        # Initialize the system
        self._load_data()
        self._calculate_indicators()

    def _load_data(self):
        """Load market data for all timeframes"""
        print(f"Loading data for {self.symbol}...")

        for interval, period in self.periods.items():
            try:
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(period=period, interval=interval)

                if not df.empty:
                    # Calculate returns
                    df["Returns"] = df["Close"].pct_change()
                    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

                    self.data[interval] = df
                    print(f"✓ Loaded {len(df)} bars for {interval} timeframe")
                else:
                    print(f"✗ No data available for {interval} timeframe")

            except Exception as e:
                print(f"✗ Error loading {interval} data: {e}")

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for regime detection"""
        df = df.copy()

        # Moving Averages
        df["EMA_Fast"] = df["Close"].ewm(span=self.ma_fast).mean()
        df["EMA_Slow"] = df["Close"].ewm(span=self.ma_slow).mean()
        df["MA_Diff"] = df["EMA_Fast"] - df["EMA_Slow"]
        df["MA_Signal"] = (df["EMA_Fast"] > df["EMA_Slow"]).astype(int)

        # Volatility Indicators
        if TALIB_AVAILABLE:
            df["ATR"] = talib.ATR(
                df["High"], df["Low"], df["Close"], timeperiod=self.atr_period
            )
        else:
            # Custom ATR calculation
            df["TR"] = np.maximum(
                df["High"] - df["Low"],
                np.maximum(
                    abs(df["High"] - df["Close"].shift(1)),
                    abs(df["Low"] - df["Close"].shift(1)),
                ),
            )
            df["ATR"] = df["TR"].rolling(window=self.atr_period).mean()

        df["Volatility"] = df["Returns"].rolling(
            window=self.volatility_lookback
        ).std() * np.sqrt(252)
        df["ATR_Norm"] = df["ATR"] / df["Close"] * 100

        # Bollinger Bands
        bb_period = 20
        df["BB_Middle"] = df["Close"].rolling(window=bb_period).mean()
        bb_std = df["Close"].rolling(window=bb_period).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
        df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (
            df["BB_Upper"] - df["BB_Lower"]
        )

        # RSI
        if TALIB_AVAILABLE:
            df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
        else:
            # Custom RSI calculation
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        if TALIB_AVAILABLE:
            df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = talib.MACD(df["Close"])
        else:
            # Custom MACD calculation
            ema12 = df["Close"].ewm(span=12).mean()
            ema26 = df["Close"].ewm(span=26).mean()
            df["MACD"] = ema12 - ema26
            df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
            df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # Volume indicators (if available)
        if "Volume" in df.columns:
            df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()
            df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]

        return df

    def _calculate_indicators(self):
        """Calculate indicators for all timeframes"""
        print("Calculating technical indicators...")

        for interval, df in self.data.items():
            self.indicators[interval] = self._calculate_technical_indicators(df)
            print(f"✓ Calculated indicators for {interval}")

    def _detect_regime_hmm(self, df: pd.DataFrame, n_states: int = 4) -> np.ndarray:
        """
        Detect market regimes using Hidden Markov Model approach
        Similar to Jim Simons' methodology
        """
        # Prepare features for regime detection
        features = []

        # Returns and volatility
        if "Returns" in df.columns:
            features.append(df["Returns"].fillna(0))
        if "Volatility" in df.columns:
            features.append(df["Volatility"].fillna(df["Volatility"].mean()))
        if "ATR_Norm" in df.columns:
            features.append(df["ATR_Norm"].fillna(df["ATR_Norm"].mean()))
        if "MA_Diff" in df.columns:
            features.append(df["MA_Diff"].fillna(0))

        if not features:
            return np.full(len(df), 0)

        # Combine features
        X = np.column_stack(features)

        # Remove NaN rows
        valid_mask = ~np.isnan(X).any(axis=1)
        if not np.any(valid_mask):
            return np.full(len(df), 0)

        X_clean = X[valid_mask]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Fit Gaussian Mixture Model (approximates HMM for regime detection)
        try:
            gmm = GaussianMixture(n_components=n_states, random_state=42, max_iter=100)
            gmm.fit(X_scaled)
            states_clean = gmm.predict(X_scaled)

            # Map back to original dataframe
            states = np.full(len(df), 0)
            states[valid_mask] = states_clean

            return states
        except Exception as e:
            print(f"Warning: HMM detection failed: {e}")
            return np.full(len(df), 0)

    def _classify_regime(self, df: pd.DataFrame, idx: int = -1) -> MarketRegime:
        """Classify current market regime based on trend and volatility"""
        if len(df) < 50:  # Need minimum data
            return MarketRegime.UNKNOWN

        current = df.iloc[idx]
        recent = df.iloc[max(0, idx - 20) : idx + 1]

        # Determine trend direction
        ma_signal = current.get("MA_Signal", 0)
        ma_diff = current.get("MA_Diff", 0)

        # Calculate trend strength
        trend_consistency = (
            recent["MA_Signal"].mean() if "MA_Signal" in recent.columns else 0.5
        )

        # Determine volatility level
        current_vol = current.get("ATR_Norm", 0)
        avg_vol = (
            recent["ATR_Norm"].mean() if "ATR_Norm" in recent.columns else current_vol
        )
        vol_threshold = avg_vol * 1.2  # 20% above average = high volatility

        is_trending_up = ma_signal > 0 and trend_consistency > 0.6
        is_trending_down = ma_signal <= 0 and trend_consistency < 0.4
        is_sideways = 0.4 <= trend_consistency <= 0.6
        is_volatile = current_vol > vol_threshold

        # Classify regime
        if is_trending_up:
            return (
                MarketRegime.BULL_VOLATILE if is_volatile else MarketRegime.BULL_QUIET
            )
        elif is_trending_down:
            return (
                MarketRegime.BEAR_VOLATILE if is_volatile else MarketRegime.BEAR_QUIET
            )
        elif is_sideways:
            return (
                MarketRegime.SIDEWAYS_VOLATILE
                if is_volatile
                else MarketRegime.SIDEWAYS_QUIET
            )
        else:
            return MarketRegime.UNKNOWN

    def _get_trading_strategy(self, regime: MarketRegime) -> TradingStrategy:
        """Get recommended trading strategy for regime"""
        strategy_map = {
            MarketRegime.BULL_QUIET: TradingStrategy.TREND_FOLLOWING,
            MarketRegime.BULL_VOLATILE: TradingStrategy.MOMENTUM,
            MarketRegime.BEAR_QUIET: TradingStrategy.TREND_FOLLOWING,  # Short bias
            MarketRegime.BEAR_VOLATILE: TradingStrategy.BREAKOUT,  # Short breakouts
            MarketRegime.SIDEWAYS_QUIET: TradingStrategy.MEAN_REVERSION,
            MarketRegime.SIDEWAYS_VOLATILE: TradingStrategy.AVOID,
            MarketRegime.UNKNOWN: TradingStrategy.DEFENSIVE,
        }
        return strategy_map.get(regime, TradingStrategy.DEFENSIVE)

    def _get_position_sizing_multiplier(
        self, regime: MarketRegime, volatility: float
    ) -> float:
        """Calculate position sizing multiplier based on regime and volatility"""
        base_multipliers = {
            MarketRegime.BULL_QUIET: 1.0,
            MarketRegime.BULL_VOLATILE: 0.7,
            MarketRegime.BEAR_QUIET: 0.8,
            MarketRegime.BEAR_VOLATILE: 0.5,
            MarketRegime.SIDEWAYS_QUIET: 0.9,
            MarketRegime.SIDEWAYS_VOLATILE: 0.3,
            MarketRegime.UNKNOWN: 0.4,
        }

        base = base_multipliers.get(regime, 0.5)

        # Adjust for volatility (inverse relationship)
        vol_adjustment = max(0.2, min(1.5, 1.0 / (1.0 + volatility * 0.1)))

        return base * vol_adjustment

    def _generate_signals(
        self, df: pd.DataFrame, regime: MarketRegime
    ) -> Tuple[List[str], List[str]]:
        """Generate entry and exit signals based on regime"""
        if len(df) < 2:
            return [], []

        current = df.iloc[-1]
        previous = df.iloc[-2]

        entry_signals = []
        exit_signals = []

        # RSI signals
        rsi = current.get("RSI", 50)
        if rsi < 30:
            entry_signals.append("RSI Oversold (Potential Buy)")
        elif rsi > 70:
            entry_signals.append("RSI Overbought (Potential Sell)")

        # MACD signals
        macd = current.get("MACD", 0)
        macd_signal = current.get("MACD_Signal", 0)
        prev_macd = previous.get("MACD", 0)
        prev_macd_signal = previous.get("MACD_Signal", 0)

        if macd > macd_signal and prev_macd <= prev_macd_signal:
            entry_signals.append("MACD Bullish Crossover")
        elif macd < macd_signal and prev_macd >= prev_macd_signal:
            entry_signals.append("MACD Bearish Crossover")

        # Moving Average signals
        if current.get("MA_Signal", 0) != previous.get("MA_Signal", 0):
            if current.get("MA_Signal", 0) > 0:
                entry_signals.append("Golden Cross (Bullish)")
            else:
                entry_signals.append("Death Cross (Bearish)")

        # Bollinger Band signals
        bb_pos = current.get("BB_Position", 0.5)
        if bb_pos <= 0.05:
            entry_signals.append("Price at Lower Bollinger Band")
        elif bb_pos >= 0.95:
            entry_signals.append("Price at Upper Bollinger Band")

        # Regime-specific signals
        if regime in [MarketRegime.BULL_QUIET, MarketRegime.BULL_VOLATILE]:
            if current.get("Close", 0) > current.get("EMA_Fast", 0):
                entry_signals.append("Price Above Fast EMA (Bullish)")
        elif regime in [MarketRegime.BEAR_QUIET, MarketRegime.BEAR_VOLATILE]:
            if current.get("Close", 0) < current.get("EMA_Fast", 0):
                entry_signals.append("Price Below Fast EMA (Bearish)")

        # Exit signals based on volatility
        atr_norm = current.get("ATR_Norm", 0)
        if atr_norm > df["ATR_Norm"].quantile(0.9):
            exit_signals.append("Extreme Volatility - Consider Exits")

        return entry_signals, exit_signals

    def _identify_key_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Identify key support/resistance levels"""
        if len(df) < 50:
            return {}

        current_price = df["Close"].iloc[-1]
        recent_data = df.tail(50)

        # Support and resistance levels
        highs = recent_data["High"]
        lows = recent_data["Low"]

        resistance = highs.quantile(0.95)
        support = lows.quantile(0.05)

        # Moving averages as dynamic levels
        ema_fast = df["EMA_Fast"].iloc[-1]
        ema_slow = df["EMA_Slow"].iloc[-1]

        # Bollinger Bands
        bb_upper = df["BB_Upper"].iloc[-1]
        bb_lower = df["BB_Lower"].iloc[-1]
        bb_middle = df["BB_Middle"].iloc[-1]

        return {
            "current_price": current_price,
            "resistance": resistance,
            "support": support,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
        }

    def analyze_current_regime(self, timeframe: str = "1D") -> RegimeAnalysis:
        """Analyze current market regime for specified timeframe"""
        if timeframe not in self.indicators:
            raise ValueError(f"Timeframe {timeframe} not available")

        df = self.indicators[timeframe]

        # Detect regime
        current_regime = self._classify_regime(df)

        # Calculate confidence based on trend consistency
        recent = df.tail(20)
        trend_consistency = (
            recent["MA_Signal"].mean() if "MA_Signal" in recent.columns else 0.5
        )
        confidence = abs(trend_consistency - 0.5) * 2  # Scale to 0-1

        # Get current values
        current = df.iloc[-1]
        trend_direction = "Bullish" if current.get("MA_Signal", 0) > 0 else "Bearish"

        current_vol = current.get("ATR_Norm", 0)
        avg_vol = (
            recent["ATR_Norm"].mean() if "ATR_Norm" in recent.columns else current_vol
        )
        volatility_level = "High" if current_vol > avg_vol * 1.2 else "Normal"

        # Get strategy recommendation
        recommended_strategy = self._get_trading_strategy(current_regime)

        # Calculate position sizing
        position_multiplier = self._get_position_sizing_multiplier(
            current_regime, current_vol
        )

        # Determine risk level
        if current_regime in [
            MarketRegime.SIDEWAYS_VOLATILE,
            MarketRegime.BEAR_VOLATILE,
        ]:
            risk_level = "High"
        elif current_regime in [MarketRegime.BULL_VOLATILE]:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Generate signals
        entry_signals, exit_signals = self._generate_signals(df, current_regime)

        # Identify key levels
        key_levels = self._identify_key_levels(df)

        return RegimeAnalysis(
            current_regime=current_regime,
            confidence=confidence,
            trend_direction=trend_direction,
            volatility_level=volatility_level,
            recommended_strategy=recommended_strategy,
            position_sizing_multiplier=position_multiplier,
            risk_level=risk_level,
            entry_signals=entry_signals,
            exit_signals=exit_signals,
            key_levels=key_levels,
        )

    def print_analysis_report(self, timeframe: str = "1D"):
        """Print comprehensive analysis report"""
        analysis = self.analyze_current_regime(timeframe)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("=" * 80)
        print(f"MARKET REGIME ANALYSIS REPORT - {self.symbol}")
        print(f"Timeframe: {timeframe} | Generated: {current_time}")
        print("=" * 80)

        print(f"\n📊 CURRENT REGIME: {analysis.current_regime.value}")
        print(f"🎯 Confidence: {analysis.confidence:.1%}")
        print(f"📈 Trend Direction: {analysis.trend_direction}")
        print(f"📉 Volatility Level: {analysis.volatility_level}")

        print(f"\n💼 TRADING RECOMMENDATIONS:")
        print(f"Strategy: {analysis.recommended_strategy.value}")
        print(
            f"Position Sizing: {analysis.position_sizing_multiplier:.1%} of normal size"
        )
        print(f"Risk Level: {analysis.risk_level}")

        print(f"\n🔑 KEY LEVELS:")
        levels = analysis.key_levels
        print(f"Current Price: ${levels.get('current_price', 0):.2f}")
        print(f"Resistance: ${levels.get('resistance', 0):.2f}")
        print(f"Support: ${levels.get('support', 0):.2f}")
        print(f"EMA Fast: ${levels.get('ema_fast', 0):.2f}")
        print(f"EMA Slow: ${levels.get('ema_slow', 0):.2f}")

        if analysis.entry_signals:
            print(f"\n🟢 ENTRY SIGNALS:")
            for signal in analysis.entry_signals:
                print(f"  • {signal}")

        if analysis.exit_signals:
            print(f"\n🔴 EXIT SIGNALS:")
            for signal in analysis.exit_signals:
                print(f"  • {signal}")

        # Strategy-specific recommendations
        print(f"\n📋 STRATEGY-SPECIFIC RECOMMENDATIONS:")
        self._print_strategy_recommendations(analysis)

        print("\n" + "=" * 80)

    def _print_strategy_recommendations(self, analysis: RegimeAnalysis):
        """Print detailed strategy recommendations"""
        strategy = analysis.recommended_strategy
        regime = analysis.current_regime

        if strategy == TradingStrategy.TREND_FOLLOWING:
            print("• Use moving average crossovers for entries")
            print("• Set stop-losses below recent swing lows (for longs)")
            print("• Trail stops as trend progresses")
            if regime in [MarketRegime.BEAR_QUIET, MarketRegime.BEAR_VOLATILE]:
                print("• Consider short positions or inverse ETFs")

        elif strategy == TradingStrategy.MOMENTUM:
            print("• Look for breakouts above resistance")
            print("• Use wider stop-losses due to volatility")
            print("• Take partial profits on quick moves")
            print("• Monitor volume for confirmation")

        elif strategy == TradingStrategy.MEAN_REVERSION:
            print("• Buy near lower Bollinger Band")
            print("• Sell near upper Bollinger Band")
            print("• Use RSI < 30 for long entries")
            print("• Use RSI > 70 for short entries")
            print("• Set tight profit targets")

        elif strategy == TradingStrategy.BREAKOUT:
            print("• Wait for clear breakout confirmation")
            print("• Use volume to confirm breakouts")
            print("• Set stops below breakout level")
            print("• Be prepared for false breakouts")

        elif strategy == TradingStrategy.AVOID:
            print("• Reduce position sizes significantly")
            print("• Avoid new positions")
            print("• Focus on risk management")
            print("• Wait for regime change")

        elif strategy == TradingStrategy.DEFENSIVE:
            print("• Use smaller position sizes")
            print("• Focus on high-quality setups only")
            print("• Maintain tight risk controls")
            print("• Consider cash or defensive assets")

    def plot_regime_analysis(self, timeframe: str = "1D", days: int = 60):
        """Create comprehensive regime analysis plot"""
        if timeframe not in self.indicators:
            print(f"No data available for timeframe {timeframe}")
            return

        df = self.indicators[timeframe].tail(days * 24 if "5m" in timeframe else days)

        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(
            f"{self.symbol} Market Regime Analysis - {timeframe}",
            fontsize=16,
            fontweight="bold",
        )

        # Plot 1: Price with moving averages and regime
        ax1 = axes[0]
        ax1.plot(df.index, df["Close"], label="Close", color="black", linewidth=1)
        ax1.plot(
            df.index,
            df["EMA_Fast"],
            label=f"EMA {self.ma_fast}",
            color="blue",
            alpha=0.7,
        )
        ax1.plot(
            df.index,
            df["EMA_Slow"],
            label=f"EMA {self.ma_slow}",
            color="red",
            alpha=0.7,
        )

        # Add Bollinger Bands
        ax1.plot(df.index, df["BB_Upper"], color="gray", alpha=0.3, linestyle="--")
        ax1.plot(df.index, df["BB_Lower"], color="gray", alpha=0.3, linestyle="--")
        ax1.fill_between(
            df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.1, color="gray"
        )

        ax1.set_title("Price Action & Moving Averages")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Volatility
        ax2 = axes[1]
        ax2.plot(df.index, df["ATR_Norm"], label="ATR %", color="purple")
        ax2.axhline(
            y=df["ATR_Norm"].mean(),
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="Average ATR",
        )
        ax2.axhline(
            y=df["ATR_Norm"].mean() * 1.2,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="High Vol Threshold",
        )
        ax2.set_title("Volatility (ATR %)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: RSI and MACD
        ax3 = axes[2]
        ax3_twin = ax3.twinx()

        # RSI
        ax3.plot(df.index, df["RSI"], label="RSI", color="green")
        ax3.axhline(y=70, color="red", linestyle="--", alpha=0.5)
        ax3.axhline(y=30, color="green", linestyle="--", alpha=0.5)
        ax3.axhline(y=50, color="gray", linestyle="-", alpha=0.3)
        ax3.set_ylabel("RSI", color="green")
        ax3.set_ylim(0, 100)

        # MACD
        ax3_twin.plot(df.index, df["MACD"], label="MACD", color="blue", alpha=0.7)
        ax3_twin.plot(
            df.index, df["MACD_Signal"], label="Signal", color="red", alpha=0.7
        )
        ax3_twin.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax3_twin.set_ylabel("MACD", color="blue")

        ax3.set_title("RSI & MACD")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Regime classification
        ax4 = axes[3]

        # Calculate regime for each point
        regimes = []
        for i in range(len(df)):
            if i < 20:  # Need minimum data
                regimes.append(0)
            else:
                regime = self._classify_regime(df, i)
                regime_map = {
                    MarketRegime.BULL_QUIET: 1,
                    MarketRegime.BULL_VOLATILE: 2,
                    MarketRegime.SIDEWAYS_QUIET: 3,
                    MarketRegime.SIDEWAYS_VOLATILE: 4,
                    MarketRegime.BEAR_QUIET: 5,
                    MarketRegime.BEAR_VOLATILE: 6,
                    MarketRegime.UNKNOWN: 0,
                }
                regimes.append(regime_map.get(regime, 0))

        # Create color map
        colors = [
            "gray",
            "lightgreen",
            "green",
            "yellow",
            "orange",
            "lightcoral",
            "red",
        ]
        regime_names = [
            "Unknown",
            "Bull Quiet",
            "Bull Volatile",
            "Sideways Quiet",
            "Sideways Volatile",
            "Bear Quiet",
            "Bear Volatile",
        ]

        ax4.scatter(
            df.index, regimes, c=regimes, cmap=ListedColormap(colors), alpha=0.7, s=10
        )
        ax4.set_yticks(range(7))
        ax4.set_yticklabels(regime_names, fontsize=8)
        ax4.set_title("Market Regime Classification")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_continuous_monitoring(self, update_interval: int = 300):
        """Run continuous monitoring with specified update interval (seconds)"""
        print(f"Starting continuous monitoring for {self.symbol}")
        print(f"Update interval: {update_interval} seconds")
        print("Press Ctrl+C to stop")

        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")  # Clear screen

                # Reload data
                self._load_data()
                self._calculate_indicators()

                # Print analysis for main timeframes
                for tf in ["1D", "1H", "15m"]:
                    if tf in self.indicators:
                        print(f"\n{'='*20} {tf} TIMEFRAME {'='*20}")
                        self.print_analysis_report(tf)

                print(f"\nNext update in {update_interval} seconds...")
                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")

    def export_analysis_to_csv(self, filename: str = None):
        """Export detailed analysis to CSV file"""
        if filename is None:
            filename = f"{self.symbol}_regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        results = []

        for timeframe in self.indicators.keys():
            try:
                analysis = self.analyze_current_regime(timeframe)
                df = self.indicators[timeframe]
                current = df.iloc[-1]

                result = {
                    "timestamp": datetime.now(),
                    "symbol": self.symbol,
                    "timeframe": timeframe,
                    "regime": analysis.current_regime.value,
                    "confidence": analysis.confidence,
                    "trend_direction": analysis.trend_direction,
                    "volatility_level": analysis.volatility_level,
                    "recommended_strategy": analysis.recommended_strategy.value,
                    "position_sizing_multiplier": analysis.position_sizing_multiplier,
                    "risk_level": analysis.risk_level,
                    "current_price": analysis.key_levels.get("current_price", 0),
                    "resistance": analysis.key_levels.get("resistance", 0),
                    "support": analysis.key_levels.get("support", 0),
                    "rsi": current.get("RSI", 0),
                    "macd": current.get("MACD", 0),
                    "atr_norm": current.get("ATR_Norm", 0),
                    "entry_signals": "; ".join(analysis.entry_signals),
                    "exit_signals": "; ".join(analysis.exit_signals),
                }
                results.append(result)

            except Exception as e:
                print(f"Error analyzing {timeframe}: {e}")

        df_results = pd.DataFrame(results)
        df_results.to_csv(filename, index=False)
        print(f"Analysis exported to {filename}")


def main():
    """Main function to demonstrate the Market Regime Analyzer"""
    print("Market Regime Analysis System")
    print("Based on Jim Simons' Hidden Markov Model methodology")
    print("=" * 60)

    # Get user input
    symbol = input("Enter symbol (default: SPY): ").strip().upper() or "SPY"

    print(f"\nInitializing analyzer for {symbol}...")

    try:
        # Initialize analyzer
        analyzer = MarketRegimeAnalyzer(symbol=symbol)

        while True:
            print("\n" + "=" * 60)
            print("MARKET REGIME ANALYZER - MAIN MENU")
            print("=" * 60)
            print("1. Current Regime Analysis (All Timeframes)")
            print("2. Detailed Analysis (Single Timeframe)")
            print("3. Generate Charts")
            print("4. Export Analysis to CSV")
            print("5. Start Continuous Monitoring")
            print("6. Multi-Symbol Analysis")
            print("7. Exit")

            choice = input("\nSelect option (1-7): ").strip()

            if choice == "1":
                print("\nAnalyzing all timeframes...")
                for tf in ["1D", "1H", "15m", "5m"]:
                    if tf in analyzer.indicators:
                        analyzer.print_analysis_report(tf)
                        print()

            elif choice == "2":
                print("\nAvailable timeframes:", list(analyzer.indicators.keys()))
                tf = input("Enter timeframe (default: 1D): ").strip() or "1D"
                if tf in analyzer.indicators:
                    analyzer.print_analysis_report(tf)
                else:
                    print(f"Timeframe {tf} not available")

            elif choice == "3":
                print("\nAvailable timeframes:", list(analyzer.indicators.keys()))
                tf = input("Enter timeframe for charts (default: 1D): ").strip() or "1D"
                days = input(
                    "Enter number of periods to display (default: 60): "
                ).strip()
                days = int(days) if days.isdigit() else 60

                if tf in analyzer.indicators:
                    analyzer.plot_regime_analysis(tf, days)
                else:
                    print(f"Timeframe {tf} not available")

            elif choice == "4":
                filename = input("Enter filename (optional): ").strip() or None
                analyzer.export_analysis_to_csv(filename)

            elif choice == "5":
                interval = input("Update interval in seconds (default: 300): ").strip()
                interval = int(interval) if interval.isdigit() else 300
                analyzer.run_continuous_monitoring(interval)

            elif choice == "6":
                symbols_input = input(
                    "Enter symbols separated by commas (e.g., SPY,QQQ,IWM): "
                ).strip()
                if symbols_input:
                    symbols = [s.strip().upper() for s in symbols_input.split(",")]
                    print(f"\nAnalyzing multiple symbols: {symbols}")

                    for sym in symbols:
                        try:
                            print(f"\n{'='*80}")
                            print(f"ANALYSIS FOR {sym}")
                            print("=" * 80)

                            temp_analyzer = MarketRegimeAnalyzer(symbol=sym)
                            temp_analyzer.print_analysis_report("1D")

                        except Exception as e:
                            print(f"Error analyzing {sym}: {e}")
                else:
                    print("No symbols entered")

            elif choice == "7":
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please select 1-7.")

            input("\nPress Enter to continue...")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have an internet connection and the symbol exists.")


# Additional utility functions


class PortfolioRegimeAnalyzer:
    """Analyze multiple assets for portfolio-level regime insights"""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.analyzers = {}

        for symbol in symbols:
            try:
                self.analyzers[symbol] = MarketRegimeAnalyzer(symbol)
                print(f"✓ Loaded {symbol}")
            except Exception as e:
                print(f"✗ Failed to load {symbol}: {e}")

    def get_portfolio_regime_summary(self) -> Dict:
        """Get portfolio-level regime summary"""
        regime_counts = {}
        strategy_counts = {}
        avg_volatility = 0
        total_symbols = 0

        for symbol, analyzer in self.analyzers.items():
            try:
                analysis = analyzer.analyze_current_regime("1D")

                # Count regimes
                regime = analysis.current_regime.value
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

                # Count strategies
                strategy = analysis.recommended_strategy.value
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                # Average volatility
                df = analyzer.indicators["1D"]
                vol = df["ATR_Norm"].iloc[-1]
                avg_volatility += vol
                total_symbols += 1

            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")

        if total_symbols > 0:
            avg_volatility /= total_symbols

        return {
            "regime_distribution": regime_counts,
            "strategy_distribution": strategy_counts,
            "average_volatility": avg_volatility,
            "total_symbols": total_symbols,
        }

    def print_portfolio_summary(self):
        """Print portfolio regime summary"""
        summary = self.get_portfolio_regime_summary()

        print("=" * 80)
        print("PORTFOLIO REGIME ANALYSIS")
        print("=" * 80)

        print(f"Symbols Analyzed: {summary['total_symbols']}")
        print(f"Average Volatility: {summary['average_volatility']:.2f}%")

        print("\nRegime Distribution:")
        for regime, count in summary["regime_distribution"].items():
            pct = count / summary["total_symbols"] * 100
            print(f"  {regime}: {count} symbols ({pct:.1f}%)")

        print("\nStrategy Distribution:")
        for strategy, count in summary["strategy_distribution"].items():
            pct = count / summary["total_symbols"] * 100
            print(f"  {strategy}: {count} symbols ({pct:.1f}%)")

        # Portfolio-level recommendations
        print("\nPortfolio Recommendations:")

        # Dominant regime
        if summary["regime_distribution"]:
            dominant_regime = max(
                summary["regime_distribution"].items(), key=lambda x: x[1]
            )
            print(
                f"• Dominant regime: {dominant_regime[0]} ({dominant_regime[1]} symbols)"
            )

        # Risk assessment
        volatile_regimes = ["Bull Volatile", "Bear Volatile", "Sideways Volatile"]
        volatile_count = sum(
            summary["regime_distribution"].get(r, 0) for r in volatile_regimes
        )
        volatile_pct = (
            volatile_count / summary["total_symbols"] * 100
            if summary["total_symbols"] > 0
            else 0
        )

        if volatile_pct > 50:
            print("• HIGH RISK: Majority of symbols in volatile regimes")
            print("• Recommendation: Reduce overall position sizes")
        elif volatile_pct > 25:
            print("• MEDIUM RISK: Significant volatility detected")
            print("• Recommendation: Use defensive position sizing")
        else:
            print("• LOW RISK: Most symbols in stable regimes")
            print("• Recommendation: Normal position sizing acceptable")


# Risk Management Calculator
class RiskCalculator:
    """Calculate position sizes and risk metrics"""

    @staticmethod
    def calculate_position_size(
        account_size: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss: float,
        regime_multiplier: float = 1.0,
    ) -> Dict:
        """Calculate optimal position size"""
        risk_amount = account_size * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share == 0:
            return {"error": "Invalid stop loss level"}

        base_shares = risk_amount / risk_per_share
        adjusted_shares = base_shares * regime_multiplier
        position_value = adjusted_shares * entry_price

        return {
            "shares": int(adjusted_shares),
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_per_share": risk_per_share,
            "regime_adjustment": regime_multiplier,
            "position_percent": position_value / account_size * 100,
        }

    @staticmethod
    def calculate_kelly_criterion(
        win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        if avg_loss == 0:
            return 0

        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Cap at 25% for safety
        return max(0, min(0.25, kelly_fraction))


if __name__ == "__main__":
    main()
