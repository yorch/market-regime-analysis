# AGENTS.md

Guidelines for AI code assistants (Claude Code, Gemini, etc.) working in this repository.

## Project Overview

Market regime analysis system using HMMs to classify market states and generate trading signals. See [README.md](README.md) for user-facing overview and [docs/status.md](docs/status.md) for current project state.

**Multi-package uv workspace** with three packages:
- **mra_lib** — Core library (zero UI/framework deps)
- **mra_cli** — CLI interface (depends on mra_lib)
- **mra_web** — FastAPI web API (depends on mra_lib)

## Development Commands

### Environment Setup

```bash
# Install all dependencies (including dev tools)
uv sync

# Or using just
just install
```

### Running the Application

```bash
# CLI — get help
uv run mra --help

# CLI — current regime analysis
uv run mra current-analysis --symbol SPY --provider yfinance

# API server
uv run mra-api
uv run mra-api --dev

# Optimization
uv run mra-optimize --mode grid --symbol SPY --provider yfinance

# Dev runner (no install needed)
python run.py --help
```

### CLI Commands

The CLI uses Click with these key commands:

```bash
uv run mra current-analysis --symbol SPY --provider alphavantage
uv run mra detailed-analysis --symbol SPY --timeframe 1D --provider alphavantage
uv run mra generate-charts --symbol SPY --timeframe 1D --days 60
uv run mra multi-symbol-analysis --symbols "SPY,QQQ,IWM" --timeframe 1D
uv run mra position-sizing --base-size 0.02 --regime "Bull Trending" --confidence 0.8
uv run mra export-csv --symbol SPY --filename analysis.csv
uv run mra continuous-monitoring --symbol SPY --interval 300
uv run mra regime-forecast --symbol SPY --steps 10 --timeframe 1D
uv run mra calibrate-multipliers --symbol SPY --method sharpe_weighted
uv run mra list-providers
uv run mra start-api --dev
```

### Code Quality

```bash
# Using just (preferred)
just qa          # fmt + lint + types — gate before commit
just fmt         # Format code
just lint        # Lint code
just fix         # Fix auto-fixable issues
just types       # Type check with mypy

# Or directly
uv run ruff format packages/ examples/
uv run ruff check packages/ examples/
uv run ruff check --fix packages/ examples/
uv run mypy packages/
```

### Testing

```bash
# All tests
just test        # or: uv run pytest

# Unit tests only (exclude integration/slow)
just test-unit

# Per-package
just test-lib    # or: uv run pytest packages/mra_lib/tests/
just test-cli
just test-web

# Specific test files
uv run pytest packages/mra_lib/tests/test_strategy.py -v
uv run pytest packages/mra_lib/tests/test_engine.py -v
uv run pytest packages/mra_lib/tests/test_forecasting.py -v
uv run pytest packages/mra_lib/tests/test_calibrator.py -v
uv run pytest packages/mra_lib/tests/test_transaction_costs.py -v
uv run pytest packages/mra_lib/tests/test_providers.py -v
uv run pytest packages/mra_lib/tests/test_risk_calculator.py -v
uv run pytest packages/mra_lib/tests/test_system.py -v
```

### Docker

```bash
just docker-build    # Build image
just docker-up       # Build and run
just docker-down     # Stop
```

## Architecture Overview

### Workspace Structure

```bash
market-regime-analysis/
├── packages/
│   ├── mra_lib/                    # Core library (zero UI deps)
│   │   ├── pyproject.toml
│   │   ├── src/mra_lib/
│   │   │   ├── __init__.py         # Re-exports all public API
│   │   │   ├── analyzer.py         # MarketRegimeAnalyzer — main orchestrator
│   │   │   ├── config/             # Enums, data classes, settings
│   │   │   │   ├── enums.py        # MarketRegime, TradingStrategy
│   │   │   │   └── data_classes.py # RegimeAnalysis dataclass
│   │   │   ├── types/              # Protocol definitions
│   │   │   │   └── protocols.py    # DashboardProtocol, DataStoreProtocol, etc.
│   │   │   ├── indicators/         # HMM-based detectors
│   │   │   │   ├── hmm_detector.py       # GMM-based HMM
│   │   │   │   └── true_hmm_detector.py  # hmmlearn-based HMM
│   │   │   ├── data_providers/     # Plug-and-play provider architecture
│   │   │   │   ├── base.py         # MarketDataProvider ABC + registry
│   │   │   │   ├── alphavantage_provider.py
│   │   │   │   ├── polygon_provider.py
│   │   │   │   ├── yfinance_provider.py
│   │   │   │   └── mock_provider.py
│   │   │   ├── backtesting/        # Strategy optimization framework
│   │   │   │   ├── engine.py       # BacktestEngine
│   │   │   │   ├── strategy.py     # RegimeStrategy
│   │   │   │   ├── walk_forward.py # WalkForwardValidator
│   │   │   │   ├── optimizer.py    # StrategyOptimizer
│   │   │   │   ├── metrics.py      # PerformanceMetrics
│   │   │   │   ├── calibrator.py   # RegimeMultiplierCalibrator
│   │   │   │   └── transaction_costs.py
│   │   │   ├── risk/               # Risk management
│   │   │   │   └── risk_calculator.py  # SimonsRiskCalculator + PortfolioPositionLimits
│   │   │   └── portfolio/          # Multi-asset analysis
│   │   │       └── portfolio.py    # PortfolioHMMAnalyzer
│   │   └── tests/                  # All lib tests
│   ├── mra_cli/                    # CLI package
│   │   ├── pyproject.toml
│   │   ├── src/mra_cli/
│   │   │   ├── main.py             # Click CLI commands
│   │   │   └── optimization.py     # Strategy optimization runner
│   │   └── tests/
│   └── mra_web/                    # Web API package
│       ├── pyproject.toml
│       ├── src/mra_web/
│       │   ├── app.py              # FastAPI application
│       │   ├── config.py           # API configuration (Pydantic)
│       │   ├── server.py           # Uvicorn startup script
│       │   ├── auth.py             # JWT + API key auth
│       │   ├── endpoints.py        # API route handlers
│       │   ├── models.py           # Pydantic request/response models
│       │   ├── utils.py            # JSON encoding, metrics
│       │   └── websocket.py        # WebSocket support
│       └── tests/
├── pyproject.toml                  # Workspace root (tool config)
├── Justfile                        # Task runner
├── Dockerfile                      # Two-stage Docker build
├── docker-compose.yml
├── docker-compose.postgres.yml     # Optional Postgres override
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── .env.example
├── run.py                          # Dev runner (no install needed)
├── examples/
├── docs/
└── uv.lock
```

### Key Architectural Principle

The core library (`mra_lib`) has **zero knowledge of UI frameworks**:
- `mra_lib` defines Protocol classes in `types/protocols.py` (e.g., `DashboardProtocol`, `DataStoreProtocol`, `MarketDataProviderProtocol`)
- CLI and web packages each implement those protocols
- New frontends can be added without touching core logic

### Import Conventions

```python
# From external code (CLI, web, tests):
from mra_lib import MarketRegimeAnalyzer, MarketRegime
from mra_lib.data_providers import MarketDataProvider
from mra_lib.backtesting import RegimeStrategy, BacktestEngine
from mra_lib.indicators.true_hmm_detector import TrueHMMDetector
from mra_lib.config.enums import MarketRegime, TradingStrategy
from mra_lib.risk.risk_calculator import SimonsRiskCalculator

# Within mra_lib, use absolute imports:
from mra_lib.config.enums import MarketRegime
from mra_lib.data_providers import MarketDataProvider

# Within the same sub-package, relative imports are fine:
from .base import MarketDataProvider  # inside data_providers/
from .strategy import RegimeStrategy  # inside backtesting/
```

### Key Components

1. **MarketRegimeAnalyzer** (`analyzer.py`): Central analysis engine that coordinates data fetching, regime detection, and reporting
2. **HiddenMarkovRegimeDetector** (`indicators/hmm_detector.py`): Core HMM implementation using Gaussian Mixture Models with 6-state regime classification
3. **TrueHMMDetector** (`indicators/true_hmm_detector.py`): Full HMM implementation using hmmlearn with Viterbi decoding, regime forecasting, and stability analysis
4. **Data Providers** (`data_providers/`): Plug-and-play architecture supporting Alpha Vantage, Polygon.io, and Yahoo Finance
5. **Portfolio Analysis** (`portfolio/portfolio.py`): Multi-symbol correlation and regime analysis
6. **Risk Management** (`risk/risk_calculator.py`): Kelly Criterion-based position sizing with regime adjustments
7. **Backtester** (`backtesting/`): Walk-forward validation and strategy optimization framework

### Data Flow

1. Data Provider fetches market data (daily/hourly/15min timeframes)
2. Feature engineering: returns, volatility, skewness, kurtosis, autocorrelation
3. HMM analysis using Gaussian Mixture Models for regime classification
4. Statistical arbitrage signal generation (mean reversion, momentum breakdown)
5. Risk-adjusted position sizing using Kelly Criterion
6. Comprehensive reporting and visualization

## Tooling Stack

| Tool | Purpose | Config location |
|------|---------|-----------------|
| uv | Package manager & workspace | `pyproject.toml` + `uv.lock` |
| Ruff | Linting + formatting | `[tool.ruff]` in root `pyproject.toml` |
| mypy | Type checking (with Pydantic plugin) | `[tool.mypy]` in root `pyproject.toml` |
| pytest | Testing (with pytest-asyncio) | `[tool.pytest.ini_options]` in root `pyproject.toml` |
| pre-commit | Git hooks (ruff + mypy) | `.pre-commit-config.yaml` |
| just | Task runner | `Justfile` |

## Development Guidelines

### Code Style
- Uses Ruff for formatting and linting with 100-character line length
- Python 3.13+, 4-space indentation, type hints required throughout
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Comprehensive docstrings following numpy/scipy style

### Testing Strategy
- Tests live alongside each package: `packages/<pkg>/tests/`
- Root pytest config collects from all package test directories
- Markers: `integration` (external APIs), `slow`
- **Minimum coverage: 65%** — enforced in CI via `--cov-fail-under=65` and in `pyproject.toml` `[tool.coverage.report]`
- Run `just test-cov` to verify coverage locally; CI will fail if coverage drops below the threshold
- Conventions: name tests `test_<unit>_<behavior>()`; use fixtures and deterministic inputs
- Run `just qa` before committing (fmt + lint + types)

### Commit & PR Guidelines
- Commits follow Conventional Commits: `feat:`, `fix:`, `docs:`, etc.
- Write imperative, scoped messages: `feat(cli): add multi-symbol analysis`
- PRs must include: clear description, test evidence, and impact notes
- Pass CI (lint + tests) and keep diffs focused; update docs for user-facing changes

### Adding New Packages
1. Create `packages/mra_<name>/` with `pyproject.toml` and `src/mra_<name>/`
2. Add `mra-<name>` to root `pyproject.toml` workspace members (already `packages/*`)
3. Implement protocols from `mra_lib.types.protocols`

### Adding New Data Providers
1. Implement the `MarketDataProvider` base class in `mra_lib/data_providers/`
2. Register in the package `__init__.py`

### Modifying the Backtester
1. **Adding strategy parameters**: Add to `RegimeStrategy.__init__()`, expose in `from_param_vector()`
2. **Adding cost models**: Subclass `TransactionCostModel` in `backtesting/transaction_costs.py`
3. **Modifying walk-forward**: Adjust parameters in `WalkForwardValidator`

### Security & Configuration
- Use environment variables for secrets: `ALPHA_VANTAGE_API_KEY`, `POLYGON_API_KEY`, `JWT_SECRET`
- Avoid `--api-key` in shell history; use `export VAR=...` or `.env`
- CORS/rate limits/JWT configured via `config.py`/env; review before exposing the API

## CI Pipeline

Four jobs run in parallel, Docker depends on all:
1. **lint** — ruff check + ruff format --check
2. **typecheck** — mypy (soft-fail until fully annotated)
3. **test** — pytest across all packages
4. **build** — uv build to verify packages build
5. **docker** — multi-stage build on main/tags

## Dependencies

Python 3.13+ required. All deps managed via `uv` with workspace support — see `pyproject.toml` files.
