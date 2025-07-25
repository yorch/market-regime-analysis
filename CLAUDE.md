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
# Interactive CLI application with all features
uv run main.py

# Direct module execution for testing
uv run python main.py --help

# Basic system test
uv run test_system.py
```

### CLI Commands

The main application uses Click CLI with these key commands:

```bash
# Current regime analysis for all timeframes
uv run main.py current-analysis --symbol SPY --provider alphavantage

# Detailed single timeframe analysis
uv run main.py detailed-analysis --symbol SPY --timeframe 1D --provider alphavantage

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

# Test data providers
uv run test_alpha_vantage.py        # Alpha Vantage API test
uv run test_alpha_vantage_provider.py  # Provider integration test
uv run test_yf_aapl.py             # Yahoo Finance test (currently broken)

# Mock data testing
uv run test_mock.py

# All pytest tests
uv run pytest
```

## Architecture Overview

### Core Package Structure

```
market_regime_analysis/
├── __init__.py          # Main exports
├── analyzer.py          # MarketRegimeAnalyzer - main analysis engine
├── data_classes.py      # RegimeAnalysis dataclass
├── data_provider.py     # AlphaVantageProvider, YFinanceProvider
├── enums.py            # MarketRegime, TradingStrategy enums
├── hmm_detector.py     # HiddenMarkovRegimeDetector - HMM implementation
├── portfolio.py        # PortfolioHMMAnalyzer - multi-asset analysis
└── risk_calculator.py  # SimonsRiskCalculator - position sizing
```

### Key Components

1. **MarketRegimeAnalyzer** (`analyzer.py`): Central analysis engine that coordinates data fetching, regime detection, and reporting
2. **HiddenMarkovRegimeDetector** (`hmm_detector.py`): Core HMM implementation using Gaussian Mixture Models with 6-state regime classification
3. **Data Providers** (`data_provider.py`): Abstracted data access supporting Alpha Vantage and Yahoo Finance (with fallback due to current outages)
4. **Portfolio Analysis** (`portfolio.py`): Multi-symbol correlation and regime analysis
5. **Risk Management** (`risk_calculator.py`): Kelly Criterion-based position sizing with regime adjustments

### Data Flow

1. Data Provider fetches market data (daily/hourly/15min timeframes)
2. Feature engineering: returns, volatility, skewness, kurtosis, autocorrelation
3. HMM analysis using Gaussian Mixture Models for regime classification
4. Statistical arbitrage signal generation (mean reversion, momentum breakdown)
5. Risk-adjusted position sizing using Kelly Criterion
6. Comprehensive reporting and visualization

## Data Provider Configuration

### Alpha Vantage (Recommended)

The system defaults to Alpha Vantage due to Yahoo Finance outages (July 2025):

```bash
export ALPHA_VANTAGE_API_KEY=your_key_here
```

### Yahoo Finance (Currently Broken)

Yahoo Finance is experiencing widespread outages. Use Alpha Vantage as primary data source.

## Important Implementation Details

### HMM States and Regime Mapping

- 6-state HMM implementation using sklearn's GaussianMixture
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
- Mock data testing for offline development
- Manual verification through CLI commands

### Key Design Patterns

- Strategy pattern for data providers (Alpha Vantage/Yahoo Finance)
- Factory pattern for analyzer initialization
- Dataclass for structured analysis results
- Enum-based regime and strategy classification

## Common Development Tasks

### Adding New Market Regimes

1. Update `MarketRegime` enum in `enums.py`
2. Update regime detection logic in `hmm_detector.py`
3. Add strategy mapping in `TradingStrategy` enum
4. Update risk multipliers in `risk_calculator.py`

### Adding New Data Providers

1. Implement provider interface in `data_provider.py`
2. Add provider choice to CLI options in `main.py`
3. Update initialization logic in `analyzer.py`
4. Add integration tests

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
