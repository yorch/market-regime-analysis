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
Market Regime Analysis System (Multi-Package Workspace)
├── packages/
│   ├── mra_lib/    — Core library (zero UI deps)
│   ├── mra_cli/    — CLI interface
│   └── mra_web/    — FastAPI web API
├── pyproject.toml  — Workspace root
├── Justfile        — Task runner
└── Dockerfile      — Two-stage build
```

The project uses a **uv workspace** with three packages so that the core
analysis library (`mra_lib`) carries no UI or web dependencies. The `just`
task runner (see the `Justfile` at the repo root) provides convenient
shortcuts for common development tasks such as `just test`, `just lint`,
and `just fmt`.

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
uv run mra --help
```

1. **Run a quick analysis (no API key required, uses Yahoo Finance):**

```bash
uv run mra current-analysis --provider yfinance --symbol SPY
```

1. **Test the system:**

```bash
just test
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
- **Cross-Asset Position Limits**: Portfolio-level exposure enforcement (gross, net, per-asset, sector, max positions)

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

Run `uv run mra --help` to see all commands. Common commands:

- Current regime analysis (all timeframes): `uv run mra current-analysis --symbol SPY --provider yfinance`
- Detailed single timeframe analysis: `uv run mra detailed-analysis --symbol SPY --timeframe 1D --provider alphavantage --api-key YOUR_KEY`
- Generate charts (5 panels): `uv run mra generate-charts --symbol SPY --timeframe 1D --days 60 --provider yfinance`
- Export analysis to CSV: `uv run mra export-csv --symbol SPY --filename analysis.csv --provider yfinance`
- Continuous monitoring: `uv run mra continuous-monitoring --symbol SPY --interval 300 --provider alphavantage --api-key YOUR_KEY`
- Multi-symbol portfolio analysis: `uv run mra multi-symbol-analysis --symbols "SPY,QQQ,IWM" --timeframe 1D --provider yfinance`
- Position sizing calculator: `uv run mra position-sizing --base-size 0.02 --regime "Bull Trending" --confidence 0.8 --persistence 0.7 --correlation 0.0`
- List available data providers and capabilities: `uv run mra list-providers`
- Strategy optimization (grid search): `uv run mra-optimize --mode grid --symbol SPY --provider yfinance`
- Strategy optimization (random search): `uv run mra-optimize --mode random --symbol SPY --provider yfinance --iterations 50`

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

### Core Classes (`mra_lib`)

- **`MarketRegimeAnalyzer`** (`mra_lib.analyzer`): Main analysis engine
- **`HiddenMarkovRegimeDetector`** (`mra_lib.indicators.hmm_detector`): GMM-based HMM implementation
- **`TrueHMMDetector`** (`mra_lib.indicators.true_hmm_detector`): hmmlearn-based HMM with Viterbi decoding (used by backtester)
- **`PortfolioHMMAnalyzer`** (`mra_lib.portfolio`): Multi-asset analysis
- **`SimonsRiskCalculator`** (`mra_lib.risk_calculator`): Risk management utilities
- **`PortfolioPositionLimits`** (`mra_lib.risk_calculator`): Cross-asset exposure limit enforcement

### Backtester Classes (`mra_lib.backtesting`)

- **`BacktestEngine`** (`mra_lib.backtesting.engine`): Trade simulation with transaction costs, stop-loss/take-profit, and portfolio position limits
- **`RegimeStrategy`** (`mra_lib.backtesting.strategy`): Parameterized trading strategy with `from_param_vector()` for optimization
- **`WalkForwardValidator`** (`mra_lib.backtesting.walk_forward`): Out-of-sample walk-forward validation framework
- **`StrategyOptimizer`** (`mra_lib.backtesting.optimizer`): Grid/random search over strategy parameters
- **`PerformanceMetrics`** (`mra_lib.backtesting.metrics`): Comprehensive performance statistics and Kelly Criterion
- **`TransactionCostModel`** (`mra_lib.backtesting.transaction_costs`): Configurable cost models (equity, futures, retail, HFT)
- **`RegimeMultiplierCalibrator`** (`mra_lib.backtesting.calibrator`): Empirical calibration of regime multipliers from backtest data

### Data Classes (`mra_lib`)

- **`MarketRegime`** (`mra_lib.enums`): Enum for regime classifications
- **`TradingStrategy`** (`mra_lib.enums`): Enum for strategy recommendations
- **`RegimeAnalysis`** (`mra_lib.data_classes`): Comprehensive analysis results

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning (Gaussian Mixture Models)
- **hmmlearn**: Hidden Markov Model implementation (Viterbi decoding)
- **click**: CLI framework
- **yfinance**: Market data retrieval (free)
- **alpha-vantage**: Market data retrieval (API key required)
- **polygon-api-client**: Market data retrieval (API key required)
- **matplotlib**: Visualization and charting

Python: 3.13+

## 🧪 Testing

The preferred way to run the full test suite is via the task runner:

```bash
just test
```

You can also run tests directly with pytest:

```bash
uv run pytest
```

Or run a specific test file:

```bash
uv run pytest packages/mra_lib/tests/test_system.py
```

The tests cover:

- Analyzer initialization and regime detection (`test_system.py`, `test_mock.py`)
- RegimeStrategy signal generation, parameter vectors, confidence scaling (`test_strategy.py`)
- BacktestEngine direction propagation, LONG/SHORT entries, direction reversals (`test_engine.py`)
- Walk-forward return compounding and window aggregation (`test_engine.py`)
- Optimizer scoring, ranking, and print robustness (`test_engine.py`)
- Regime forecasting: n-step projection, stationary distribution, stability (`test_forecasting.py`)
- Regime multiplier calibration: scoring methods, normalization (`test_calibrator.py`)
- Transaction cost models: all components, presets, P&L after costs (`test_transaction_costs.py`)
- Provider base class, registry, factory, mock provider, DataFrame standardization (`test_providers.py`)
- SimonsRiskCalculator, PortfolioPositionLimits, BacktestEngine integration (`test_risk_calculator.py`)

## 🔧 Configuration

The system can be configured through various parameters:

```python
# Custom periods for different timeframes
periods = {
    "1D": "2y",    # Daily data for 2 years
    "1H": "6mo",   # Hourly data for 6 months
    "15m": "2mo"   # 15-min data for 2 months
}

from mra_lib import MarketRegimeAnalyzer
analyzer = MarketRegimeAnalyzer("SPY", periods=periods)
```

### Data Providers and API Keys

- Default CLI provider: Alpha Vantage (set `ALPHA_VANTAGE_API_KEY` or pass `--api-key`).
- Other providers: Yahoo Finance (free): `--provider yfinance` (no API key); Polygon.io (pro): set `POLYGON_API_KEY` or pass `--api-key` and use `--provider polygon`.

List providers and their capabilities:

```bash
uv run mra list-providers
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
