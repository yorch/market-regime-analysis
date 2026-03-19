# Market Regime Analysis

[![CI](https://github.com/yorch/market-regime-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/yorch/market-regime-analysis/actions/workflows/ci.yml)

A Python system for detecting market regimes using Hidden Markov Models. It classifies market states (Bull/Bear Trending, Mean Reverting, High/Low Volatility, Breakout), generates statistical arbitrage signals, and provides risk-adjusted position sizing.

> **Disclaimer**: This system is for educational and research purposes. The current strategy does not outperform buy-and-hold. Do not deploy with real capital without independent validation. See [docs/status.md](docs/status.md) for a candid assessment.

## Architecture

```text
packages/
├── mra_lib/    — Core library (zero UI deps)
├── mra_cli/    — CLI interface (Click)
└── mra_web/    — REST API (FastAPI)
```

The project uses a **uv workspace** so the core analysis library carries no UI or web dependencies. The `just` task runner provides shortcuts for common tasks.

## Quick Start

```bash
# Install
git clone <repository-url>
cd market-regime-analysis
uv sync

# Run analysis (no API key needed)
uv run mra current-analysis --provider yfinance --symbol SPY

# Run tests
just test
```

See `uv run mra --help` for all CLI commands.

## Core Features

- **HMM Regime Detection** — 6-state classification using `hmmlearn` (Baum-Welch training, Viterbi decoding) and a GMM-based detector
- **Statistical Arbitrage Signals** — Z-score mean reversion, autocorrelation breakdown
- **Risk Management** — Kelly Criterion position sizing with regime adjustments and portfolio-level exposure limits
- **Backtesting** — Walk-forward validation, transaction cost modeling, grid/random strategy optimization
- **Multi-Timeframe** — Daily, hourly, 15-minute analysis
- **3 Data Providers** — Yahoo Finance (free), Alpha Vantage, Polygon.io
- **REST API** — FastAPI with JWT auth, WebSocket monitoring ([docs/api.md](docs/api.md))

### Example Output

```text
================================================================================
HMM MARKET REGIME ANALYSIS - SPY (1D)
================================================================================
Current Price: $432.50
Analysis Time: 2025-07-19 15:30:00

REGIME CLASSIFICATION:
   Current Regime: Bull Trending
   HMM State: 2
   Confidence: 82.5%
   Persistence: 75.0%
   Transition Prob: 68.2%

TRADING RECOMMENDATION:
   Strategy: Trend Following
   Position Size: 15.8%
   Risk Level: Medium

STATISTICAL ARBITRAGE:
   Mean Reversion: LONG signal (Z-score: -2.15)
   Momentum Breakdown: Low autocorr (0.045)

KEY LEVELS:
   RESISTANCE: $438.20
   SUPPORT: $425.80
   SMA_50: $429.15
   BB_UPPER: $441.50
   BB_LOWER: $423.70
================================================================================
```

## CLI Commands

```bash
uv run mra current-analysis --symbol SPY --provider yfinance
uv run mra detailed-analysis --symbol SPY --timeframe 1D
uv run mra generate-charts --symbol SPY --timeframe 1D --days 60
uv run mra multi-symbol-analysis --symbols "SPY,QQQ,IWM"
uv run mra position-sizing --base-size 0.02 --regime "Bull Trending" --confidence 0.8
uv run mra export-csv --symbol SPY --filename analysis.csv
uv run mra regime-forecast --symbol SPY --steps 10
uv run mra calibrate-multipliers --symbol SPY --method sharpe_weighted
uv run mra list-providers
uv run mra start-api --dev
uv run mra-optimize --mode grid --symbol SPY --provider yfinance
```

## Data Providers

| Provider | API Key | Setup |
|----------|---------|-------|
| Yahoo Finance | Not required | `--provider yfinance` |
| Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | `--provider alphavantage` |
| Polygon.io | `POLYGON_API_KEY` | `--provider polygon` |

## Mathematical Approach

### Hidden Markov Models

The system provides two HMM implementations:
- **GMM-based detector** (`hmm_detector.py`): Gaussian Mixture Models as HMM approximation with post-hoc transition matrix estimation
- **True HMM detector** (`true_hmm_detector.py`): Full `hmmlearn` implementation with Baum-Welch training and Viterbi decoding

### Statistical Features

- **Returns & Log Returns**: Basic price movement analysis
- **Volatility Measures**: Rolling standard deviation, ATR
- **Higher-Order Moments**: Skewness and kurtosis for distribution analysis
- **Autocorrelation**: Momentum persistence indicators (lags 1, 2, 5)
- **Cross-Correlations**: Feature interaction analysis (return-vol ratio, trend-vol)

### Risk Management

- **Kelly Criterion**: `f* = (bp - q) / b` with confidence scaling
- **Regime Multipliers**: Risk adjustments based on market conditions
- **Correlation Adjustments**: Portfolio diversification considerations
- **Safety Caps**: Maximum position limits (1% min, 50% max)
- **Portfolio Limits**: Gross/net exposure, per-asset, sector, and max positions

## Configuration

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

## Dependencies

- **pandas**, **numpy** — Data manipulation and numerical computing
- **scikit-learn** — Gaussian Mixture Models
- **hmmlearn** — True HMM implementation (Viterbi decoding)
- **click** — CLI framework
- **yfinance**, **alpha-vantage**, **polygon-api-client** — Market data providers
- **matplotlib** — Visualization
- **fastapi**, **uvicorn**, **pydantic** — Web API
- **python-jose**, **slowapi**, **websockets** — Auth, rate limiting, WebSocket

Python 3.13+ required. All deps managed via `uv`.

## Testing

```bash
just test        # All tests
uv run pytest    # Or directly with pytest
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

## Contributing

Contributions are welcome. Please ensure:

- Type hints throughout
- Docstrings following numpy/scipy style
- Unit tests for new functionality
- `just qa` passes before submitting

## Development

```bash
just qa          # Format + lint + type-check (run before committing)
just test        # All tests
just test-unit   # Unit tests only (no integration/slow)
just test-lib    # Core library tests only
```

See [AGENTS.md](AGENTS.md) for full development guide, architecture details, and contribution guidelines.

## Documentation

| Document | Description |
|----------|-------------|
| [AGENTS.md](AGENTS.md) | Development guide, architecture, tooling |
| [docs/api.md](docs/api.md) | REST API reference |
| [docs/status.md](docs/status.md) | Current project state and known limitations |
| [docs/archive/](docs/archive/) | Historical planning and review documents |

## License

Educational and research purposes. Past performance does not guarantee future results.
