# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a professional-grade market regime analysis system implementing Jim Simons' Hidden Markov Model methodology for quantitative trading analysis. The system detects market regimes (Bull Trending, Bear Trending, Mean Reverting, High/Low Volatility, Breakout) and provides statistical arbitrage signals following Renaissance Technologies' approach.

## Development Commands

### Environment Setup

```bash
# Install dependencies
uv sync

# Activate virtual environment (if needed)
source venv/bin/activate
```

### Running the Application

```bash
# Get help menu
uv run main.py --help

# Basic system test
uv run test_system.py
```

### CLI Commands

The main application uses Click CLI with these key commands:

```bash
# Current regime analysis for all timeframes
uv run main.py current-analysis --symbol SPY --provider alphavantage

# Detailed single timeframe analysis (with different providers)
uv run main.py detailed-analysis --symbol SPY --timeframe 1D --provider alphavantage
uv run main.py detailed-analysis --symbol SPY --timeframe 1D --provider polygon

# Generate HMM visualization charts
uv run main.py generate-charts --symbol SPY --timeframe 1D --days 60

# Multi-symbol portfolio analysis
uv run main.py multi-symbol-analysis --symbols "SPY,QQQ,IWM" --timeframe 1D

# Position sizing calculator
uv run main.py position-sizing --base-size 0.02 --regime "Bull Trending" --confidence 0.8

# Export analysis to CSV
uv run main.py export-csv --symbol SPY --filename analysis.csv

# Continuous monitoring
uv run main.py continuous-monitoring --symbol SPY --interval 300

# Regime forecasting (uses HMM transition matrix)
uv run main.py regime-forecast --symbol SPY --steps 10 --timeframe 1D
uv run main.py regime-forecast --symbol SPY --steps 5 --n-states 4

# Empirical regime multiplier calibration
uv run main.py calibrate-multipliers --symbol SPY --method sharpe_weighted
uv run main.py calibrate-multipliers --symbol SPY --method kelly --output calibration.json
```

### Strategy Optimization

```bash
# Grid search over strategy parameters with walk-forward validation
uv run run_optimization.py --mode grid --symbol SPY --provider yfinance

# Random search with custom iteration count
uv run run_optimization.py --mode random --symbol SPY --provider yfinance --iterations 50

# Baseline (default parameters) only
uv run run_optimization.py --mode baseline --symbol SPY --provider yfinance

# Custom output path for results
uv run run_optimization.py --mode grid --output results/my_optimization.json
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Fix auto-fixable issues
uv run ruff check --fix

# Run tests
uv run pytest
```

### Testing

```bash
# Basic system functionality test
uv run test_system.py

# Mock data testing
uv run test_mock.py

# RegimeStrategy unit tests (get_signal, generate_signals, from_param_vector)
uv run pytest test_strategy.py -v

# BacktestEngine, walk-forward, and optimizer tests
uv run pytest test_engine.py -v

# Regime forecasting unit tests
uv run pytest test_forecasting.py -v

# Regime multiplier calibration unit tests
uv run pytest test_calibrator.py -v

# All pytest tests
uv run pytest
```

## Architecture Overview

### Core Package Structure

```bash
market_regime_analysis/
├── __init__.py              # Main exports
├── analyzer.py              # MarketRegimeAnalyzer - main analysis engine
├── data_classes.py          # RegimeAnalysis dataclass
├── enums.py                 # MarketRegime, TradingStrategy enums
├── hmm_detector.py          # HiddenMarkovRegimeDetector - GMM-based HMM
├── true_hmm_detector.py     # TrueHMMDetector - hmmlearn-based HMM
├── portfolio.py             # PortfolioHMMAnalyzer - multi-asset analysis
├── risk_calculator.py       # SimonsRiskCalculator - position sizing
├── backtester/              # Backtesting and strategy optimization
│   ├── __init__.py          # Package exports
│   ├── engine.py            # BacktestEngine - trade simulation
│   ├── strategy.py          # RegimeStrategy - parameterized trading strategy
│   ├── walk_forward.py      # WalkForwardValidator - out-of-sample testing
│   ├── optimizer.py         # StrategyOptimizer - grid/random search
│   ├── metrics.py           # PerformanceMetrics - Sharpe, drawdown, Kelly
│   └── transaction_costs.py # Cost models (equity, futures, retail, HFT)
└── providers/               # Plug-and-play data provider architecture
    ├── __init__.py          # Auto-discovery and registration
    ├── base.py              # MarketDataProvider base class
    ├── alphavantage_provider.py  # Alpha Vantage implementation
    ├── polygon_provider.py       # Polygon.io implementation
    ├── yfinance_provider.py      # Yahoo Finance implementation
    └── mock_provider.py          # Mock provider for testing
```

### Key Components

1. **MarketRegimeAnalyzer** (`analyzer.py`): Central analysis engine that coordinates data fetching, regime detection, and reporting
2. **HiddenMarkovRegimeDetector** (`hmm_detector.py`): Core HMM implementation using Gaussian Mixture Models with 6-state regime classification
3. **TrueHMMDetector** (`true_hmm_detector.py`): Full HMM implementation using hmmlearn with Viterbi decoding, regime forecasting via transition matrix projection, and regime stability analysis (stationary distribution, expected durations)
4. **Data Providers** (`providers/`): Plug-and-play architecture supporting Alpha Vantage, Polygon.io, and Yahoo Finance with automatic provider discovery
5. **Portfolio Analysis** (`portfolio.py`): Multi-symbol correlation and regime analysis
6. **Risk Management** (`risk_calculator.py`): Kelly Criterion-based position sizing with regime adjustments
7. **Backtester** (`backtester/`): Walk-forward validation and strategy optimization framework:
   - **BacktestEngine** (`engine.py`): Simulates trades with transaction costs, stop-loss/take-profit, and LONG/SHORT support
   - **RegimeStrategy** (`strategy.py`): Parameterized strategy mapping regimes to directions and position sizes (tunable via `from_param_vector()`)
   - **WalkForwardValidator** (`walk_forward.py`): Anchored/rolling walk-forward with periodic HMM retraining on truly out-of-sample windows
   - **StrategyOptimizer** (`optimizer.py`): Grid and random search over strategy parameters, scored by composite Sharpe/excess-return metric
   - **PerformanceMetrics** (`metrics.py`): Sharpe, Sortino, Calmar, drawdown, win rate, profit factor, Kelly Criterion parameters
   - **Transaction Costs** (`transaction_costs.py`): Configurable cost models (equity, futures, retail, HFT) with spread, commission, slippage, and market impact
   - **RegimeMultiplierCalibrator** (`calibrator.py`): Empirical calibration of regime multipliers from walk-forward backtest data, with four scoring methods (Sharpe-weighted, win rate, profit factor, Kelly)

### Provider Architecture

The system uses a plug-and-play provider architecture with:

- **Abstract Base Class**: `MarketDataProvider` defines the interface for all providers
- **Auto-Discovery**: Providers are automatically registered on import
- **Factory Pattern**: `MarketDataProvider.create_provider()` creates provider instances
- **Extensibility**: Easy to add new providers by implementing the base interface

### Data Flow

1. Data Provider fetches market data (daily/hourly/15min timeframes)
2. Feature engineering: returns, volatility, skewness, kurtosis, autocorrelation
3. HMM analysis using Gaussian Mixture Models for regime classification
4. Statistical arbitrage signal generation (mean reversion, momentum breakdown)
5. Risk-adjusted position sizing using Kelly Criterion
6. Comprehensive reporting and visualization

### Backtester Data Flow

1. Walk-forward validator splits data into train/test windows
2. TrueHMMDetector trains on training window, predicts regimes on test window
3. RegimeStrategy maps regimes + confidences to directions and position sizes (scaled by `base_position_fraction`)
4. BacktestEngine simulates trades with explicit directions, transaction costs, and stop-loss/take-profit
5. PerformanceMetrics computes Sharpe, drawdown, Kelly parameters per window
6. Results are compounded across windows and aggregated

## Data Provider Configuration

The system now supports three data providers with automatic provider detection and plug-and-play architecture:

### Alpha Vantage (Default)

Professional-grade data provider with good coverage:

```bash
export ALPHA_VANTAGE_API_KEY=your_key_here
uv run main.py current-analysis --provider alphavantage --symbol SPY
```

### Polygon.io (Professional)

High-quality institutional-grade data with tick-level precision:

```bash
export POLYGON_API_KEY=your_key_here
uv run main.py current-analysis --provider polygon --symbol SPY
```

### Yahoo Finance (Free but Limited)

Free data provider experiencing intermittent outages:

```bash
uv run main.py current-analysis --provider yfinance --symbol SPY
```

### Provider Comparison

| Provider | API Key | Rate Limit | Data Quality | Cost |
|----------|---------|------------|--------------|------|
| Alpha Vantage | Required | 5 req/min | Professional | Free tier available |
| Polygon.io | Required | 60 req/min | Institutional | Premium (basic tier available) |
| Yahoo Finance | None | 60 req/min | Community | Free |

### List Available Providers

```bash
uv run main.py list-providers
```

## Important Implementation Details

### HMM States and Regime Mapping

- Two HMM implementations:
  - `HiddenMarkovRegimeDetector` (`hmm_detector.py`): Uses sklearn's GaussianMixture (6 states), used by the main analyzer
  - `TrueHMMDetector` (`true_hmm_detector.py`): Uses hmmlearn's GaussianHMM with Viterbi decoding, used by the backtester walk-forward validator
- Statistical features: returns, volatility, skewness, kurtosis, autocorrelation
- Regime classification based on state characteristics and transition probabilities

### Multi-Timeframe Analysis

- **1D (Daily)**: 2 years of data for long-term regime trends
- **1H (Hourly)**: 6 months of data for medium-term shifts
- **15m (15-minute)**: 2 months of data for short-term changes

### Risk Management Implementation

- Kelly Criterion: f* = (bp - q) / b with confidence scaling
- Regime-specific position multipliers
- Portfolio correlation adjustments
- Safety caps: 1% minimum, 50% maximum position sizes

## Development Guidelines

### Code Style

- Uses Ruff for formatting and linting with 100-character line length
- Type hints required throughout
- Comprehensive docstrings following numpy/scipy style
- Error handling for edge cases (network failures, missing data)

### Testing Strategy

- Basic functionality tests in `test_system.py`
- Data provider integration tests
- Mock data testing for offline development in `test_mock.py`
- `test_strategy.py`: Unit tests for `RegimeStrategy` (signal generation, parameter vectors, confidence scaling, base position fraction)
- `test_engine.py`: Unit tests for `BacktestEngine` (direction propagation, LONG/SHORT entries, direction reversals), walk-forward aggregation (compounding, window win rate), optimizer scoring/ranking, and `print_top_results` robustness
- `test_forecasting.py`: Unit tests for `TrueHMMDetector` forecasting (n-step probability projection, regime sequence forecasting, stationary distribution, regime stability metrics)
- `test_calibrator.py`: Unit tests for `RegimeMultiplierCalibrator` (per-regime stats computation, scoring methods, normalization, full integration calibration)
- Manual verification through CLI commands

### Key Design Patterns

- Strategy pattern for data providers (Alpha Vantage/Yahoo Finance)
- Factory pattern for analyzer initialization
- Dataclass for structured analysis results
- Enum-based regime and strategy classification
- Parameterized strategy pattern (`RegimeStrategy`) with `from_param_vector()` for optimizer integration
- Walk-forward validation for out-of-sample backtesting
- Composite scoring for multi-objective optimization (Sharpe, excess return, drawdown, trade count)

## Common Development Tasks

### Adding New Market Regimes

1. Update `MarketRegime` enum in `enums.py`
2. Update regime detection logic in `hmm_detector.py`
3. Add strategy mapping in `TradingStrategy` enum
4. Update risk multipliers in `risk_calculator.py`

### Adding New Data Providers

1. Implement provider interface in `providers/` package
2. Add provider choice to CLI options in `main.py`
3. Update initialization logic in `analyzer.py`
4. Add integration tests

### Modifying the Backtester

1. **Adding strategy parameters**: Add to `RegimeStrategy.__init__()`, expose in `from_param_vector()`, add to optimizer grids in `optimizer.py` and `run_optimization.py`
2. **Changing position sizing**: `RegimeStrategy.get_signal()` computes the position multiplier (includes `base_position_fraction` and optional confidence scaling). The engine's `_calculate_position_size()` receives this as `position_mult` directly.
3. **Adding cost models**: Subclass `TransactionCostModel` in `transaction_costs.py`
4. **Modifying walk-forward windows**: Adjust `min_train_days`, `test_days`, `retrain_frequency`, or `anchored` in `WalkForwardValidator`
5. **Changing the optimizer scoring**: Modify `OptimizationResult.score` property in `optimizer.py`

### Extending Analysis Features

1. Add feature calculation methods to `hmm_detector.py`
2. Update `RegimeAnalysis` dataclass if new outputs needed
3. Modify reporting methods in `analyzer.py`
4. Update CLI commands if user-facing

## Dependencies and Version Management

- Uses `uv` for dependency management with lock file
- Python 3.13+ required
- Key dependencies: pandas, numpy, scikit-learn, yfinance, alpha-vantage, click, matplotlib
- Development dependencies: ruff, pytest, pre-commit

The system is designed for production quantitative trading analysis with emphasis on statistical rigor, risk management, and Renaissance Technologies' methodologies.
