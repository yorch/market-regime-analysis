# AGENTS.md

Guidelines for AI code assistants (Claude Code, Gemini, Copilot, etc.) working in this repository.
`CLAUDE.md` is a symlink to this file.

## Project Overview

Market regime analysis system using HMMs to classify market states and generate trading signals. See [README.md](README.md) for the user-facing overview and [docs/status.md](docs/status.md) for current project state and known limitations.

**Multi-package uv workspace** with three packages:
- **mra_lib** — Core library (no UI/web framework deps)
- **mra_cli** — CLI interface (depends on mra_lib; `start-api` also needs mra_web)
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

# CLI — current regime analysis (yfinance is the default provider; mock is offline)
uv run mra current-analysis --symbol SPY
uv run mra current-analysis --symbol SPY --provider mock

# API server (production by default: needs JWT_SECRET >= 32 chars)
uv run mra-api
uv run mra-api --dev                # ENVIRONMENT=development, no JWT_SECRET needed
uv run mra-token --sub alice        # mint a JWT signed with JWT_SECRET

# Optimization
uv run mra-optimize --mode grid --symbol SPY --provider yfinance

# Dev runner (imports from source; dependencies must be installed)
uv run python run.py --help
```

### CLI Commands

The CLI uses Click. `--provider` / `--api-key` may be given before or after the subcommand.

```bash
uv run mra current-analysis --symbol SPY --provider alphavantage
uv run mra detailed-analysis --symbol SPY --timeframe 1D
uv run mra generate-charts --symbol SPY --timeframe 1D --days 60 --output spy.png
uv run mra multi-symbol-analysis --symbols "SPY,QQQ,IWM" --timeframe 1D
uv run mra position-sizing --base-size 0.02 --regime "Bull Trending" --confidence 0.8
uv run mra export-csv --symbol SPY --filename analysis.csv
uv run mra continuous-monitoring --symbol SPY --interval 300   # --once / --max-iterations N
uv run mra regime-forecast --symbol SPY --steps 10 --timeframe 1D
uv run mra calibrate-multipliers --symbol SPY --method sharpe_weighted
uv run mra backtest --symbol SPY --mode walk-forward --params best.json --output trades.csv
uv run mra list-providers
uv run mra start-api --dev
```

### Code Quality

```bash
# Using just (preferred)
just qa          # fmt-check + lint + types (no file changes) — gate before commit
just fmt         # Format code
just lint        # Lint code
just fix         # Fix auto-fixable lint issues and format
just types       # Type check with mypy

# Or directly
uv run ruff format packages/ examples/
uv run ruff check packages/ examples/
uv run ruff check --fix packages/ examples/
uv run mypy packages/
```

### Testing

```bash
# All tests (includes `integration` tests that call live provider APIs)
just test        # or: uv run pytest

# Unit tests only (exclude integration/slow)
just test-unit

# Unit tests with coverage (what CI runs)
just test-cov

# Per-package
just test-lib    # or: uv run pytest packages/mra_lib/tests/
just test-cli
just test-web

# Specific test files
uv run pytest packages/mra_lib/tests/test_strategy.py -v
uv run pytest packages/mra_lib/tests/test_engine.py -v
uv run pytest packages/mra_lib/tests/test_provider_contracts.py -v
uv run pytest packages/mra_web/tests/test_web_security.py -v
```

### Docker

```bash
just docker-build    # Build image (docker compose build)
just docker-up       # Start detached (needs JWT_SECRET in .env)
just docker-health   # GET /health
just docker-down     # Stop
```

## Architecture Overview

### Workspace Structure

```bash
market-regime-analysis/
├── packages/
│   ├── mra_lib/                    # Core library (no CLI/web framework deps)
│   │   ├── pyproject.toml
│   │   ├── src/mra_lib/
│   │   │   ├── __init__.py         # Re-exports the public API
│   │   │   ├── analyzer.py         # MarketRegimeAnalyzer — main orchestrator
│   │   │   ├── config/
│   │   │   │   ├── enums.py        # MarketRegime, TradingStrategy, DirectionalBias
│   │   │   │   ├── data_classes.py # RegimeAnalysis dataclass
│   │   │   │   ├── regime_tables.py # Shared regime multipliers / strategy / bias tables
│   │   │   │   └── timeframes.py   # TIMEFRAMES, DEFAULT_PERIODS, CONFIRMATION_* defaults
│   │   │   ├── errors.py           # MRAError hierarchy (DataLoadError, ProviderError, ...)
│   │   │   ├── indicators/         # Regime model (one detector everywhere)
│   │   │   │   ├── base.py               # RegimeDetector protocol, persistence/transition helpers
│   │   │   │   ├── features.py           # Causal, scale-free feature helpers + build_hmm_features
│   │   │   │   ├── regime_mapping.py     # State -> regime decision tree (RegimeThresholds)
│   │   │   │   ├── true_hmm_detector.py  # TrueHMMDetector (hmmlearn), select_n_states (BIC)
│   │   │   │   └── hmm_detector.py       # Deprecated alias of TrueHMMDetector
│   │   │   ├── data_providers/     # Plug-and-play provider architecture
│   │   │   │   ├── base.py         # MarketDataProvider ABC, registry, error types, rate limiter
│   │   │   │   ├── _http.py        # Shared HTTP client (timeouts, retries, key redaction)
│   │   │   │   ├── credentials.py  # Provider API-key env var resolution
│   │   │   │   ├── alphavantage_provider.py
│   │   │   │   ├── polygon_provider.py
│   │   │   │   ├── alpaca_provider.py
│   │   │   │   ├── tiingo_provider.py
│   │   │   │   ├── yfinance_provider.py
│   │   │   │   └── mock_provider.py      # Offline deterministic data
│   │   │   ├── backtesting/        # Strategy optimization framework
│   │   │   │   ├── engine.py       # BacktestEngine
│   │   │   │   ├── strategy.py     # RegimeStrategy
│   │   │   │   ├── walk_forward.py # WalkForwardValidator
│   │   │   │   ├── optimizer.py    # StrategyOptimizer
│   │   │   │   ├── runner.py       # run_backtest -> BacktestReport (`mra backtest`)
│   │   │   │   ├── metrics.py      # PerformanceMetrics
│   │   │   │   ├── trade_stats.py  # Shared trade statistics
│   │   │   │   ├── calibrator.py   # RegimeMultiplierCalibrator
│   │   │   │   └── transaction_costs.py
│   │   │   ├── risk/
│   │   │   │   └── risk_calculator.py  # SimonsRiskCalculator + PortfolioPositionLimits
│   │   │   ├── signals/
│   │   │   │   └── confirmation.py # Multi-timeframe confirmation (pure confirm_timeframes)
│   │   │   └── portfolio/
│   │   │       └── portfolio.py    # PortfolioHMMAnalyzer
│   │   └── tests/
│   ├── mra_cli/
│   │   ├── pyproject.toml
│   │   ├── src/mra_cli/
│   │   │   ├── main.py             # Click CLI commands (`mra`)
│   │   │   └── optimization.py     # Strategy optimization runner (`mra-optimize`)
│   │   └── tests/
│   └── mra_web/
│       ├── pyproject.toml
│       ├── src/mra_web/
│       │   ├── app.py              # FastAPI application factory (create_app)
│       │   ├── config.py           # APIConfig, read from env (fails closed)
│       │   ├── server.py           # `mra-api` entry point; serve() shared with `mra start-api`
│       │   ├── auth.py             # JWT + X-API-Key auth; `mra-token` entry point
│       │   ├── endpoints.py        # /api/v1 route handlers
│       │   ├── errors.py           # Uniform error envelope handlers
│       │   ├── ratelimit.py        # Path-based per-client rate limiting middleware
│       │   ├── security.py         # Secret scrubbing for logs
│       │   ├── models.py           # Pydantic request/response models
│       │   ├── utils.py            # Strict JSON, thread pool + concurrency cap, metrics
│       │   └── websocket.py        # Authenticated WebSocket monitoring
│       └── tests/
├── pyproject.toml                  # Workspace root (tool config)
├── Justfile                        # Task runner
├── Dockerfile                      # Two-stage Docker build (non-root, HEALTHCHECK)
├── docker-compose.yml
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── .github/dependabot.yml
├── .env.example
├── run.py                          # Dev runner (source imports)
├── examples/                       # api_client.py, programmatic.py, custom_provider.py
├── docs/                           # api.md, status.md, archive/
└── uv.lock
```

### Key Architectural Principle

The core library (`mra_lib`) has **no dependency on UI or web frameworks**: the CLI and web packages depend on it, never the other way round.

- **No `print()`** (ruff `T201` is enforced for `mra_lib`): progress and diagnostics go to
  `logging.getLogger(__name__)` with %-style args (`G` rules); the `mra_lib` package logger has a
  `NullHandler`. Frontends configure logging (`mra` shows `mra_lib` INFO records on stderr as
  plain progress lines; `mra-optimize` writes them to stdout).
- **Reports are strings**: `format_analysis_report`, `format_confirmation_report`,
  `format_portfolio_summary`,
  `format_results`, `format_summary`, `format_top_results`, `CalibrationResult.format_report`;
  frontends echo them. The old `print_*` methods are deprecated shims (`DeprecationWarning`).
- **Typed errors** (`mra_lib/errors.py`): everything derives from `MRAError` and keeps its
  historical builtin base (`DataLoadError`/`InsufficientDataError`/`ModelNotFittedError` are
  `ValueError`s). Provider errors (`InvalidSymbolError`, `AuthError`, `RateLimitError`) propagate
  unchanged out of `MarketRegimeAnalyzer`, so CLI/web map them by type, not by cause-chain walking.
  Don't swallow-and-print; raise, or log (`logger.exception`/`warning`) and return an explicit
  result (e.g. `PortfolioHMMAnalyzer.failed_symbols`). Blind `except Exception` needs a
  `# noqa: BLE001 - reason`.

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

1. **MarketRegimeAnalyzer** (`analyzer.py`): Coordinates data fetching, feature engineering, regime detection, signals, reports and charts
2. **TrueHMMDetector** (`indicators/true_hmm_detector.py`): the single regime model, used by the analyzer, `regime-forecast`, walk-forward validation and the optimizer. Gaussian HMM (hmmlearn) on six causal, stationary features from `indicators/features.py`; diagonal covariances, `min_covar`, `n_init` EM restarts, a minimum-sample check scaled to the parameter count, volatility-sorted (canonical) state order, filtered (forward-only) posteriors for per-bar history, and absolute-threshold regime labels (`indicators/regime_mapping.py`). `MarketRegimeAnalyzer(detector_factory=...)` accepts any `RegimeDetector` (`indicators/base.py`). `HiddenMarkovRegimeDetector` is a deprecated alias (the GMM model was removed)
3. **Feature helpers** (`indicators/features.py`): pure, causal functions (ATR, rolling/log volatility, vectorized autocorrelation, z-scores, trend strength, per-bar volume ratio) shared by the detector and the analyzer's display indicators
4. **Data Providers** (`data_providers/`): Yahoo Finance, Alpha Vantage, Polygon.io, Alpaca, Tiingo and an offline mock provider
5. **Portfolio Analysis** (`portfolio/portfolio.py`): Multi-symbol regimes, returns-based correlations, Engle-Granger pairs
6. **Risk Management** (`risk/risk_calculator.py`): Kelly Criterion-based position sizing with regime and correlation adjustments
7. **Backtester** (`backtesting/`): Walk-forward validation, optimizer with holdout split, calibrator
8. **Multi-timeframe confirmation** (`signals/confirmation.py`): pure `confirm_timeframes(analyses)` -> frozen `TimeframeConfirmation` (direction, agreement, confirmed, aligned/conflicting timeframes). Bias map `REGIME_BIAS` in `config/regime_tables.py`; weights/threshold (`CONFIRMATION_WEIGHTS`, `CONFIRMATION_THRESHOLD`) in `config/timeframes.py`. Used by `current-analysis` and `POST /api/v1/analysis/confirmation`

### Data Flow

1. Data provider fetches market data (daily/hourly/15min timeframes)
2. Feature engineering (`indicators/features.py`): log return, log volatility, volatility expansion, trend strength, autocorrelation, volume ratio
3. Regime classification with `TrueHMMDetector` (the same model walk-forward validates)
4. Statistical arbitrage signal generation (mean reversion, momentum breakdown)
5. Risk-adjusted position sizing
6. Reporting, CSV export and charts

## Tooling Stack

| Tool | Purpose | Config location |
|------|---------|-----------------|
| uv | Package manager & workspace | `pyproject.toml` + `uv.lock` |
| Ruff | Linting + formatting | `[tool.ruff]` in root `pyproject.toml` |
| mypy | Type checking (with Pydantic plugin) | `[tool.mypy]` in root `pyproject.toml` |
| pytest | Testing (with pytest-asyncio) | `[tool.pytest.ini_options]` in root `pyproject.toml` |
| pre-commit | Git hooks (ruff check + format; mypy as a manual stage) | `.pre-commit-config.yaml` |
| just | Task runner | `Justfile` |
| Dependabot | Weekly uv + GitHub Actions update PRs | `.github/dependabot.yml` |

## Development Guidelines

### Code Style
- Ruff for formatting and linting, 100-character line length
- Python 3.13+, 4-space indentation, type hints throughout (mypy must stay clean)
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Google-style docstrings (`Args:`, `Returns:`, `Raises:` sections)

### Testing Strategy
- Tests live alongside each package: `packages/<pkg>/tests/`
- Root pytest config collects from all package test directories (`--strict-markers`)
- Markers: `integration` (live external APIs; excluded from CI's unit job), `slow`
- Use the `mock` provider for offline, deterministic data
- **Minimum coverage: 65%** — enforced by `fail_under` in `[tool.coverage.report]` (`just test-cov`); currently about 92%
- Conventions: name tests `test_<unit>_<behavior>()`; use fixtures and deterministic inputs
- Run `just qa` and `just test-unit` before committing

### Commit & PR Guidelines
- Commits follow Conventional Commits: `feat:`, `fix:`, `docs:`, etc.
- Write imperative, scoped messages: `feat(cli): add multi-symbol analysis`
- PRs must include: clear description, test evidence, and impact notes
- Pass CI and keep diffs focused; update docs for user-facing changes

### Adding New Packages
1. Create `packages/mra_<name>/` with `pyproject.toml` and `src/mra_<name>/`
2. Workspace membership is automatic (`members = ["packages/*"]`); add it to `[tool.uv.sources]` if other packages depend on it
3. Depend on `mra_lib`'s public API (configure `logging` and render its `format_*` reports yourself); keep `mra_lib` free of the new package's framework

### Adding New Data Providers
1. Subclass `MarketDataProvider` in `mra_lib/data_providers/` (see `examples/custom_provider.py`)
2. Register it in the package `__init__.py`; raise `InvalidSymbolError` / `AuthError` / `RateLimitError` as appropriate
3. Add its key env vars to `data_providers/credentials.py` and `.env.example`

### Modifying the Backtester
1. **Adding strategy parameters**: Add to `RegimeStrategy.__init__()`, expose in `from_param_vector()`
2. **Adding cost models**: Subclass `TransactionCostModel` in `backtesting/transaction_costs.py`
3. **Modifying walk-forward**: Adjust parameters in `WalkForwardValidator`
4. Conventions (accounting, fills, metrics) are documented in [packages/mra_lib/README.md](packages/mra_lib/README.md)

### Security & Configuration
- User-facing environment variables are listed in [.env.example](.env.example) (everything there is read by code); docker compose only forwards the ones named in `docker-compose.yml`
- Provider secrets: `ALPHA_VANTAGE_API_KEY` (or `ALPHAVANTAGE_API_KEY`), `POLYGON_API_KEY`, `APCA_API_KEY_ID` + `APCA_API_SECRET_KEY`, `TIINGO_API_KEY`
- Web API: `ENVIRONMENT` (default `production`), `JWT_SECRET` (required outside development, 32+ chars), `API_KEYS`, `CORS_ORIGINS`, `RATE_LIMIT_PER_MINUTE`, `API_TIMEOUT`, `API_MAX_CONCURRENT_ANALYSES`, `WS_MAX_CONNECTIONS`, `WS_MAX_CONNECTIONS_PER_IP` — see [docs/api.md](docs/api.md)
- CLI: `DEFAULT_PROVIDER`
- Avoid `--api-key` in shell history; use `export VAR=...` or `.env`
- Never return exception text to API clients (it can carry provider keys); map errors in `endpoints.classify_exception`

## CI Pipeline

`.github/workflows/ci.yml` (least-privilege `permissions: contents: read`, actions pinned to SHAs, `uv sync --locked`):

1. **lint** — `just lint` + `just fmt-check`
2. **typecheck** — `just types` (mypy, **blocking**)
3. **test** — `just test-cov` (unit tests with coverage; fails under 65%)
4. **build** — `uv build` for each package
5. **integration-test** — live provider APIs; manual `workflow_dispatch` only, the only job with provider secrets
6. **docker** — needs lint/typecheck/test/build; builds the image, smoke-tests `/health` and `mra --help`, pushes on main/tags

Dependabot (`.github/dependabot.yml`) opens weekly grouped PRs for uv dependencies and GitHub Actions.

## Dependencies

Python 3.13+ required. All deps managed via `uv` with workspace support — see the `pyproject.toml` files.
