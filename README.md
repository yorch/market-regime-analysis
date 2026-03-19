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
