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

# CLI — current regime analysis (yfinance is the default provider)
uv run mra current-analysis --symbol SPY

# CLI — offline, deterministic synthetic data (no network, no keys)
uv run mra current-analysis --symbol SPY --provider mock

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
uv run mra generate-charts --symbol SPY --timeframe 1D --days 60 --output spy.png
uv run mra multi-symbol-analysis --symbols "SPY,QQQ,IWM" --timeframe 1D
uv run mra position-sizing --base-size 0.02 --regime "Bull Trending" --confidence 0.8
uv run mra export-csv --symbol SPY --filename analysis.csv
uv run mra continuous-monitoring --symbol SPY --interval 300 --max-iterations 12
uv run mra regime-forecast --symbol SPY --steps 10 --timeframe 1D
uv run mra calibrate-multipliers --symbol SPY --method sharpe_weighted
uv run mra list-providers
uv run mra start-api --dev
```

CLI conventions (`mra_cli/main.py`):
- `--provider` / `--api-key` are accepted on the group and on every data subcommand (shared
  `provider_options` decorator; subcommand wins). Choices come from the provider registry.
  Resolution: subcommand > group > `$DEFAULT_PROVIDER` > `yfinance`.
- API keys are resolved lazily (`resolve_provider`) only in commands that fetch data.
- Errors go through `handle_exceptions` → exit code 1 (`--debug` re-raises with traceback).
  Commands verify library side effects (chart figure produced, CSV written) because some lib
  methods swallow errors.
- Timeframes and default periods come from `mra_lib.config.timeframes`
  (`TIMEFRAMES`, `DEFAULT_PERIODS`, `TIMEFRAME_INTERVALS`, `BARS_PER_DAY`) — don't redefine them.
- CLI tests use `CliRunner` with `--provider mock` (`packages/mra_cli/tests/test_cli_commands.py`).

### Code Quality

```bash
# Using just (preferred)
just qa          # fmt-check + lint + types — non-mutating gate before commit
just fix         # Apply ruff autofixes + format (mutates files)
just fmt         # Format code
just lint        # Lint code
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

# Unit tests with coverage (threshold = [tool.coverage.report] fail_under; CI runs this)
just test-cov

# Live-provider tests (@pytest.mark.integration; needs network/API keys)
just test-integration

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

The image runs the web API (`mra-api`) as non-root user `mra`, listening on 8000 inside the
container with a `/health` `HEALTHCHECK`. Compose publishes it on `127.0.0.1:${API_PORT:-8000}`
and **requires `JWT_SECRET`** (>=32 chars) — set it in `.env` (see `.env.example`); compose checks
this for every command, including `down`/`build`.

```bash
just docker-build    # Build image
just docker-up       # Start detached
just docker-health   # Container status + GET /health
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
│   │   │   │   ├── data_classes.py # RegimeAnalysis dataclass
│   │   │   │   ├── regime_tables.py # Canonical regime multipliers, regime→strategy map, bars/year
│   │   │   │   └── timeframes.py   # TIMEFRAMES, DEFAULT_PERIODS, intervals, bars/day
│   │   │   ├── types/              # Protocol definitions
│   │   │   │   └── protocols.py    # DashboardProtocol, DataStoreProtocol, etc.
│   │   │   ├── indicators/         # HMM-based detectors
│   │   │   │   ├── hmm_detector.py       # GMM-based HMM
│   │   │   │   └── true_hmm_detector.py  # hmmlearn-based HMM
│   │   │   ├── data_providers/     # Plug-and-play provider architecture
│   │   │   │   ├── base.py         # MarketDataProvider ABC, registry, DataFrame contract, errors, rate limiter
│   │   │   │   ├── credentials.py  # Provider env-var lookup (shared by CLI + web)
│   │   │   │   ├── _http.py        # Retrying JSON GET for REST providers
│   │   │   │   ├── alpaca_provider.py
│   │   │   │   ├── tiingo_provider.py
│   │   │   │   ├── alphavantage_provider.py
│   │   │   │   ├── polygon_provider.py
│   │   │   │   ├── yfinance_provider.py
│   │   │   │   └── mock_provider.py  # Registered as "mock" (offline, deterministic per symbol)
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
├── docker-compose.yml              # Web API service (loopback-published, JWT_SECRET required)
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
4. **Data Providers** (`data_providers/`): Plug-and-play architecture supporting Alpha Vantage, Polygon.io, Alpaca, Tiingo, and Yahoo Finance
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
| pre-commit | Git hooks (local hooks: `uv run ruff` / `uv run mypy`, versions from `uv.lock`) | `.pre-commit-config.yaml` |
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
- Markers: `integration` (external APIs / network), `slow`; `--strict-markers` is on, so unknown markers fail
- Any test that touches the network must be marked `@pytest.mark.integration`; unit tests use synthetic data
  (`synthetic_ohlcv` fixture in `packages/mra_lib/tests/conftest.py`)
- **Minimum coverage: 65%** — single source of truth is `fail_under` in `[tool.coverage.report]`; CI runs `just test-cov`
- Run `just test-cov` to verify coverage locally; CI will fail if coverage drops below the threshold
- Conventions: name tests `test_<unit>_<behavior>()`; use fixtures and deterministic inputs
- Run `just qa` before committing (fmt-check + lint + types; `just fix` to autofix)

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
See `examples/custom_provider.py` for a runnable template.
1. Implement the `MarketDataProvider` base class in `mra_lib/data_providers/`. REST providers should use `_http.get_json` (retries/backoff, timeouts, header auth, `throttle=self.throttle`) and `base.period_to_start`; client-library providers call `self.throttle()` before each request and pass `config.timeout`/`config.retries` to the client
2. Return `self.standardize_dataframe(df, interval)` — it enforces the contract documented in `base.py`: float64 OHLCV, sorted/de-duplicated tz-naive index, intraday bars in UTC, daily bars labeled by session date, optional `drop_incomplete_bar`
3. Raise `InvalidSymbolError` (a `ValueError`) for unknown symbols / empty results, `AuthError` / `RateLimitError` (both `ConnectionError`) for credential and quota problems, plain `ConnectionError` for network failures. Don't wrap `ValueError`s as `ConnectionError`
4. Set `rate_limit_per_minute` (and optionally `rate_limit_burst`) to drive the client-side token bucket (shared per provider + API key across instances; tests reset it via `MarketDataProvider.reset_rate_limiters()` in `packages/mra_lib/tests/conftest.py`)
5. Register in the package `__init__.py` — the CLI `--provider` choices are derived from the registry
6. If it needs credentials, add its env vars to `credentials.py` (`PROVIDER_ENV_VARS`, or `PROVIDER_ENV_PAIRS` for key ID + secret)
7. Add the name to `allowed_providers` (`mra_web/models.py`)

### Modifying the Backtester
1. **Adding strategy parameters**: Add to `RegimeStrategy.__init__()`, expose in `from_param_vector()`
2. **Adding cost models**: Subclass `TransactionCostModel` in `backtesting/transaction_costs.py`
3. **Modifying walk-forward**: Adjust parameters in `WalkForwardValidator`

### Security & Configuration
- Use environment variables for secrets: `ALPHA_VANTAGE_API_KEY`, `POLYGON_API_KEY`, `APCA_API_KEY_ID` + `APCA_API_SECRET_KEY` (Alpaca), `TIINGO_API_KEY`, `JWT_SECRET`
- Avoid `--api-key` in shell history; use `export VAR=...` or `.env`
- CORS/rate limits/JWT configured via `config.py`/env; review before exposing the API

## CI Pipeline

`.github/workflows/ci.yml` — workflow token is read-only by default (`permissions: contents: read`),
all installs use `uv sync --locked`, third-party actions are pinned to commit SHAs, and superseded PR
runs are cancelled (main/tag runs are not). Four jobs run in parallel; docker depends on them:
1. **lint** — ruff check + ruff format --check
2. **typecheck** — mypy (must pass; zero errors)
3. **test** — `just test-cov` (unit tests, coverage gate from `pyproject.toml`)
4. **build** — uv build to verify packages build
5. **docker** — builds the image, smoke-tests it (`docker run` + poll `/health`), pushes to GHCR on
   main/tags (or PRs labelled `publish-docker`, applied on the next push); only job with `packages: write`
6. **integration-test** — `workflow_dispatch` only; `just test-integration`; the only job given provider secrets

Dependabot (`.github/dependabot.yml`) opens weekly PRs for uv packages (minor/patch grouped, each
major separate) and GitHub Actions (SHA pins). uv itself and the Python base image are bumped by hand:
keep the Dockerfile's `ghcr.io/astral-sh/uv` tag and ci.yml's `setup-uv` `version:` in step.

## Dependencies

Python 3.13+ required. All deps managed via `uv` with workspace support — see `pyproject.toml` files.
