# Jim Simons Market Regime Analysis System

[![CI](https://github.com/yorch/market-regime-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/yorch/market-regime-analysis/actions/workflows/ci.yml)

A professional-grade Python application implementing Jim Simons' Hidden Markov Model methodology for market regime detection and quantitative trading analysis.

## 🎯 Overview

This system implements sophisticated market regime detection using Hidden Markov Models (HMMs) with the mathematical rigor and approach pioneered by Jim Simons at Renaissance Technologies. The system provides:

- **Multi-Timeframe Analysis**: Daily, hourly, and 15-minute regime detection
- **Statistical Arbitrage**: Core Renaissance strategy identification
- **Risk Management**: Kelly Criterion and regime-adjusted position sizing
- **Portfolio Analysis**: Multi-asset correlation and regime analysis

## 🏗️ System Architecture

```text
Market Regime Analysis System
├── Hidden Markov Model Implementation
├── Multi-Timeframe Analysis Engine
├── Statistical Arbitrage Detection
├── Risk Management Calculator
├── Portfolio Analysis Tools
├── Backtesting & Walk-Forward Validation
├── Strategy Parameter Optimization
└── Click-based CLI
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd market-regime-analysis
```

1. **Install dependencies:**

```bash
uv sync
```

### Basic Usage

1. **View CLI help (Click-based CLI):**

```bash
uv run main.py --help
```

1. **Run a quick analysis (no API key required, uses Yahoo Finance):**

```bash
uv run main.py current-analysis --provider yfinance --symbol SPY
```

1. **Test the system:**

```bash
uv run test_system.py
```

1. **Programmatic examples:**

```python
uv run examples/programmatic.py

# Or the API client examples (requires running the API server)
uv run examples/api_client.py
```

Notes:

- CLI default provider is Alpha Vantage (requires API key). Pass `--provider yfinance` to avoid API keys for quick tests.
- Programmatic default provider is Yahoo Finance. You can select Alpha Vantage or Polygon by passing `provider_flag` and `api_key`.

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

### 5. Backtesting & Strategy Optimization

- **Walk-Forward Validation**: Anchored or rolling out-of-sample testing with periodic HMM retraining
- **BacktestEngine**: Trade simulation with realistic transaction costs, stop-loss/take-profit, and LONG/SHORT support
- **RegimeStrategy**: Parameterized strategy mapping regimes to trade directions and position sizes
- **StrategyOptimizer**: Grid and random search over strategy parameters with composite scoring
- **Performance Metrics**: Sharpe, Sortino, Calmar ratios, drawdown analysis, Kelly Criterion parameters

## 🖥️ CLI Commands (Click)

Run `uv run main.py --help` to see all commands. Common commands:

- Current regime analysis (all timeframes): `uv run main.py current-analysis --symbol SPY --provider yfinance`
- Detailed single timeframe analysis: `uv run main.py detailed-analysis --symbol SPY --timeframe 1D --provider alphavantage --api-key YOUR_KEY`
- Generate charts (5 panels): `uv run main.py generate-charts --symbol SPY --timeframe 1D --days 60 --provider yfinance`
- Export analysis to CSV: `uv run main.py export-csv --symbol SPY --filename analysis.csv --provider yfinance`
- Continuous monitoring: `uv run main.py continuous-monitoring --symbol SPY --interval 300 --provider alphavantage --api-key YOUR_KEY`
- Multi-symbol portfolio analysis: `uv run main.py multi-symbol-analysis --symbols "SPY,QQQ,IWM" --timeframe 1D --provider yfinance`
- Position sizing calculator: `uv run main.py position-sizing --base-size 0.02 --regime "Bull Trending" --confidence 0.8 --persistence 0.7 --correlation 0.0`
- List available data providers and capabilities: `uv run main.py list-providers`
- Strategy optimization (grid search): `uv run run_optimization.py --mode grid --symbol SPY --provider yfinance`
- Strategy optimization (random search): `uv run run_optimization.py --mode random --symbol SPY --provider yfinance --iterations 50`

## 📈 Example Analysis Output

```text
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
- **`HiddenMarkovRegimeDetector`**: GMM-based HMM implementation
- **`TrueHMMDetector`**: hmmlearn-based HMM with Viterbi decoding (used by backtester)
- **`PortfolioHMMAnalyzer`**: Multi-asset analysis
- **`SimonsRiskCalculator`**: Risk management utilities

### Backtester Classes

- **`BacktestEngine`**: Trade simulation with transaction costs and stop-loss/take-profit
- **`RegimeStrategy`**: Parameterized trading strategy with `from_param_vector()` for optimization
- **`WalkForwardValidator`**: Out-of-sample walk-forward validation framework
- **`StrategyOptimizer`**: Grid/random search over strategy parameters
- **`PerformanceMetrics`**: Comprehensive performance statistics and Kelly Criterion
- **`TransactionCostModel`**: Configurable cost models (equity, futures, retail, HFT)

### Data Classes

- **`MarketRegime`**: Enum for regime classifications
- **`TradingStrategy`**: Enum for strategy recommendations
- **`RegimeAnalysis`**: Comprehensive analysis results

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning (Gaussian Mixture Models)
- **click**: CLI framework
- **yfinance**: Market data retrieval (free)
- **alpha-vantage**: Market data retrieval (API key required)
- **polygon-api-client**: Market data retrieval (API key required)
- **matplotlib**: Visualization and charting

Python: 3.13+

## 🧪 Testing

Run the test suite to verify system functionality:

```bash
uv run test_system.py
```

Or run all tests with pytest:

```bash
uv run pytest
```

The tests cover:

- Analyzer initialization and regime detection (`test_system.py`, `test_mock.py`)
- RegimeStrategy signal generation, parameter vectors, confidence scaling (`test_strategy.py`)
- BacktestEngine direction propagation, LONG/SHORT entries, direction reversals (`test_engine.py`)
- Walk-forward return compounding and window aggregation (`test_engine.py`)
- Optimizer scoring, ranking, and print robustness (`test_engine.py`)
- Report generation and error handling

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

### Data Providers and API Keys

- Default CLI provider: Alpha Vantage (set `ALPHA_VANTAGE_API_KEY` or pass `--api-key`).
- Other providers: Yahoo Finance (free): `--provider yfinance` (no API key); Polygon.io (pro): set `POLYGON_API_KEY` or pass `--api-key` and use `--provider polygon`.

List providers and their capabilities:

```bash
uv run main.py list-providers
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
