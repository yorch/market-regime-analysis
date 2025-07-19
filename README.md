# Jim Simons Market Regime Analysis System

A professional-grade Python application implementing Jim Simons' Hidden Markov Model methodology for market regime detection and quantitative trading analysis, following the exact approach used by Renaissance Technologies.

## 🎯 Overview

This system implements sophisticated market regime detection using Hidden Markov Models (HMMs) with the mathematical rigor and approach pioneered by Jim Simons at Renaissance Technologies. The system provides:

- **True HMM Implementation**: Gaussian Mixture Models with proper transition matrices
- **Multi-Timeframe Analysis**: Daily, hourly, and 15-minute regime detection
- **Statistical Arbitrage**: Core Renaissance strategy identification
- **Risk Management**: Kelly Criterion and regime-adjusted position sizing
- **Portfolio Analysis**: Multi-asset correlation and regime analysis

## 🏗️ System Architecture

```
Market Regime Analysis System
├── Hidden Markov Model Implementation
├── Multi-Timeframe Analysis Engine
├── Statistical Arbitrage Detection
├── Risk Management Calculator
├── Portfolio Analysis Tools
└── Interactive User Interface
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd market-regime-analysis
```

2. **Install dependencies:**

```bash
uv sync
```

### Basic Usage

1. **Run the interactive application:**

```bash
uv run main.py
```

2. **Or test the system:**

```bash
uv run test_system.py
```

3. **Or use programmatically:**

```python
from market_regime_analysis import MarketRegimeAnalyzer

# Single symbol analysis
analyzer = MarketRegimeAnalyzer("SPY")
analysis = analyzer.analyze_current_regime("1D")
analyzer.print_analysis_report("1D")

# Portfolio analysis
from market_regime_analysis import PortfolioHMMAnalyzer
portfolio = PortfolioHMMAnalyzer(["SPY", "QQQ", "IWM"])
portfolio.print_portfolio_summary()
```

## 📊 Core Features

### 1. Hidden Markov Model Regime Detection

- **6-State HMM**: Bull Trending, Bear Trending, Mean Reverting, High Volatility, Low Volatility, Breakout
- **Mathematical Features**: Returns, volatility, skewness, kurtosis, autocorrelation
- **Transition Matrices**: Proper state transition probability estimation
- **Regime Persistence**: Stability metrics for regime classification

### 2. Statistical Arbitrage (Simons' Core Strategy)

- **Z-Score Analysis**: Mean reversion signal identification
- **Autocorrelation Breakdown**: Momentum persistence analysis
- **Cross-Asset Pairs**: Statistical arbitrage opportunity detection
- **Confidence Weighting**: Signal strength based on regime confidence

### 3. Risk Management (Renaissance Approach)

- **Kelly Criterion**: Optimal position sizing with confidence scaling
- **Regime Adjustments**: Position multipliers based on market regime
- **Correlation Adjustments**: Portfolio diversification considerations
- **Volatility Targeting**: Risk-adjusted position sizing

### 4. Multi-Timeframe Analysis

- **Daily (1D)**: Long-term regime trends (2 years of data)
- **Hourly (1H)**: Medium-term regime shifts (6 months of data)
- **15-Minute (15m)**: Short-term regime changes (2 months of data)

## 🖥️ Interactive Menu System

The application provides a comprehensive menu system:

1. **Current HMM Regime Analysis (All Timeframes)**
2. **Detailed HMM Analysis (Single Timeframe)**
3. **Generate HMM Charts** - 5-panel visualization
4. **Export HMM Analysis to CSV** - Backtesting data
5. **Start Continuous HMM Monitoring** - Real-time analysis
6. **Multi-Symbol HMM Analysis** - Portfolio analysis
7. **Position Sizing Calculator** - Risk management
8. **Exit**

## 📈 Example Analysis Output

```
================================================================================
HMM MARKET REGIME ANALYSIS - SPY (1D)
================================================================================
Current Price: $432.50
Analysis Time: 2025-07-19 15:30:00

📊 REGIME CLASSIFICATION:
   Current Regime: Bull Trending
   HMM State: 2
   Confidence: 82.5%
   Persistence: 75.0%
   Transition Prob: 68.2%

📈 TRADING RECOMMENDATION:
   Strategy: Trend Following
   Position Size: 15.8%
   Risk Level: Medium

💰 STATISTICAL ARBITRAGE:
   • Mean Reversion: LONG signal (Z-score: -2.15)
   • Momentum Breakdown: Low autocorr (0.045)

📡 STATISTICAL SIGNALS:
   • EMA: Bullish crossover (LONG bias)
   • RSI: Neutral (52.3)
   • MACD: Bullish signal

🎯 KEY LEVELS:
   RESISTANCE: $438.20
   SUPPORT: $425.80
   SMA_50: $429.15
   BB_UPPER: $441.50
   BB_LOWER: $423.70
================================================================================
```

## 🔬 Mathematical Implementation

### Hidden Markov Models

The system uses Gaussian Mixture Models as HMM approximations with:

- **Feature Engineering**: Higher-order moments, cross-correlations
- **State Estimation**: Viterbi-like algorithms for regime detection
- **Transition Matrices**: Empirical transition probability estimation
- **Regime Mapping**: Statistical characteristics to market regimes

### Statistical Features

- **Returns & Log Returns**: Basic price movement analysis
- **Volatility Measures**: Rolling standard deviation, ATR
- **Higher-Order Moments**: Skewness and kurtosis for distribution analysis
- **Autocorrelation**: Momentum persistence indicators
- **Cross-Correlations**: Feature interaction analysis

### Risk Management

- **Kelly Criterion**: f* = (bp - q) / b with confidence scaling
- **Regime Multipliers**: Risk adjustments based on market conditions
- **Correlation Adjustments**: Portfolio diversification considerations
- **Safety Caps**: Maximum position limits (1% min, 50% max)

## 🏛️ Architecture Details

### Core Classes

- **`MarketRegimeAnalyzer`**: Main analysis engine
- **`HiddenMarkovRegimeDetector`**: HMM implementation
- **`PortfolioHMMAnalyzer`**: Multi-asset analysis
- **`SimonsRiskCalculator`**: Risk management utilities

### Data Classes

- **`MarketRegime`**: Enum for regime classifications
- **`TradingStrategy`**: Enum for strategy recommendations
- **`RegimeAnalysis`**: Comprehensive analysis results

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning (Gaussian Mixture Models)
- **yfinance**: Market data retrieval
- **matplotlib**: Visualization and charting

## 🧪 Testing

Run the test suite to verify system functionality:

```bash
python test_system.py
```

The test covers:

- Analyzer initialization
- Regime detection
- Report generation
- Error handling

## 🔧 Configuration

The system can be configured through various parameters:

```python
# Custom periods for different timeframes
periods = {
    "1D": "2y",    # Daily data for 2 years
    "1H": "6mo",   # Hourly data for 6 months
    "15m": "2mo"   # 15-min data for 2 months
}

analyzer = MarketRegimeAnalyzer("SPY", periods=periods)
```

## 📈 Performance Considerations

- **Memory Management**: Automatic cleanup in continuous monitoring
- **Error Handling**: Comprehensive exception handling
- **Data Validation**: Input validation and sanity checks
- **Efficiency**: Optimized feature calculations

## 🎯 Use Cases

1. **Quantitative Trading**: Regime-based strategy selection
2. **Risk Management**: Dynamic position sizing
3. **Portfolio Analysis**: Multi-asset regime correlation
4. **Market Research**: Statistical arbitrage identification
5. **Academic Research**: HMM implementation reference

## 📄 License

This project implements methodologies inspired by Renaissance Technologies and Jim Simons' approach to quantitative finance. The implementation is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please ensure:

- Comprehensive docstrings
- Type hints throughout
- Error handling for edge cases
- Unit tests for new functionality
- Following the existing code style

## 📞 Support

For questions or issues, please refer to the documentation or create an issue in the repository.

---

**Disclaimer**: This system is for educational and research purposes. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.
