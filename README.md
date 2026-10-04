# Market Regime Analysis

[![CI](https://github.com/yorch/market-regime-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/yorch/market-regime-analysis/actions/workflows/ci.yml)

A Python system for detecting market regimes using Hidden Markov Models. It classifies market states (Bull/Bear Trending, Mean Reverting, High/Low Volatility, Breakout), generates statistical arbitrage signals, and provides risk-adjusted position sizing.

> **Disclaimer**: This system is for educational and research purposes. The current strategy does not outperform buy-and-hold. Do not deploy with real capital without independent validation. See [docs/status.md](docs/status.md) for a candid assessment.

## Architecture

```text
packages/
├── mra_lib/    — Core library (no CLI/web framework deps)
├── mra_cli/    — CLI interface (Click)
└── mra_web/    — REST API (FastAPI)
```

The project uses a **uv workspace** so the core analysis library carries no CLI or web framework dependencies (it still prints reports and uses matplotlib for charts). The `just` task runner provides shortcuts for common tasks.

## Quick Start

```bash
# Install
git clone <repository-url>
cd market-regime-analysis
uv sync

# Run analysis (Yahoo Finance is the default provider; no API key needed)
uv run mra current-analysis --symbol SPY

# Run fully offline with deterministic synthetic data
uv run mra current-analysis --provider mock --symbol SPY

# Run the offline test suite
just test-unit
```

See `uv run mra --help` for all CLI commands.

## Core Features

### Hidden Markov Model Regime Detection

- **6-State HMM**: Bull Trending, Bear Trending, Mean Reverting, High Volatility, Low Volatility, Breakout
- **Mathematical Features**: Returns, volatility, skewness, kurtosis, autocorrelation
- **Transition Matrices**: Proper state transition probability estimation
- **Regime Persistence**: Stability metrics for regime classification

### Statistical Arbitrage

- **Z-Score Analysis**: Mean reversion signal identification
- **Autocorrelation Breakdown**: Momentum persistence analysis
- **Cross-Asset Pairs**: Statistical arbitrage opportunity detection
- **Confidence Weighting**: Signal strength based on regime confidence

### Risk Management

- **Kelly Criterion**: Optimal position sizing with confidence scaling
- **Regime Adjustments**: Position multipliers based on market regime
- **Correlation Adjustments**: Portfolio diversification considerations
- **Volatility Targeting**: Risk-adjusted position sizing
- **Cross-Asset Position Limits**: Portfolio-level exposure enforcement (gross, net, per-asset, sector, max positions)

### Multi-Timeframe Analysis

- **Daily (1D)**: Long-term regime trends (2 years of data)
- **Hourly (1H)**: Medium-term regime shifts (6 months of data)
- **15-Minute (15m)**: Short-term regime changes (1 month of data; Yahoo Finance only serves 60 days of 15m bars)
- **Multi-Timeframe Confirmation**: A pure library signal (`mra_lib.confirm_timeframes`) that says
  whether the timeframes agree: direction from the highest timeframe, a confidence-weighted
  agreement score (1D > 1H > 15m), and `confirmed` only when a lower timeframe backs the primary
  direction above the threshold. Shown by `current-analysis` and served at
  `POST /api/v1/analysis/confirmation`

### Backtesting & Strategy Optimization

- **Walk-Forward Validation**: Anchored or rolling out-of-sample testing with periodic HMM retraining
- **BacktestEngine**: Trade simulation with realistic transaction costs, stop-loss/take-profit, and LONG/SHORT support
- **RegimeStrategy**: Parameterized strategy mapping regimes to trade directions and position sizes
- **StrategyOptimizer**: Grid and random search over strategy parameters with composite scoring
- **Performance Metrics**: Sharpe, Sortino, Calmar ratios, drawdown analysis, Kelly Criterion parameters

### REST API

FastAPI with JWT / API-key auth, CSV and PNG exports, and WebSocket monitoring — see
[docs/api.md](docs/api.md).

```bash
# Local development: no credentials needed, docs at http://127.0.0.1:8000/docs
uv run mra-api --dev

# Production (the default): a JWT secret of 32+ characters is required
export JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
uv run mra-api
TOKEN=$(uv run mra-token --sub alice)          # mint a JWT (same JWT_SECRET)
curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "timeframe": "1D", "provider": "yfinance"}' \
  http://127.0.0.1:8000/api/v1/analysis/detailed
```

Alternatively set `API_KEYS` (comma-separated, 16+ characters each) and send one as
`X-API-Key`. There is no login endpoint. Note that API requests default to
`"provider": "alphavantage"`.

### Example Output

`uv run mra current-analysis --provider mock --symbol SPY` (deterministic synthetic data; the
1D section is shown, followed by 1H and 15m sections and, when two or more timeframes
succeed, a multi-timeframe confirmation section):

```text
================================================================================
HMM MARKET REGIME ANALYSIS - SPY (1D)
================================================================================
Current Price: $242.93
Analysis Time: 2026-10-04 00:50:20

📊 REGIME CLASSIFICATION:
   Current Regime: Mean Reverting
   HMM State: 1
   Confidence: 100.0%
   Persistence: 40.0%
   Transition Prob: 94.6%

📈 TRADING RECOMMENDATION:
   Strategy: Mean Reversion
   Position Multiplier: 0.46x
   Risk Level: High

📡 STATISTICAL SIGNALS:
   • MACD: Bearish signal

🎯 KEY LEVELS:
   RESISTANCE: $264.79
   SUPPORT: $239.24
   SMA_50: $250.24
   SMA_200: $237.06
   BB_UPPER: $253.70
   BB_LOWER: $238.07
   ATR_RESISTANCE: $247.15
   ATR_SUPPORT: $238.71
================================================================================
```

```text
================================================================================
MULTI-TIMEFRAME CONFIRMATION - SPY
================================================================================
🧭 SIGNAL:
   Direction: Neutral
   Confirmed: NO (primary_neutral)
   Agreement: 50.0% (threshold 60.0%)
   Confidence: 100.0%
   Primary Timeframe: 1D
   Aligned: 1D
   Conflicting: 15m
   Unavailable: 1H

📊 TIMEFRAMES:
   1D: Mean Reverting (neutral, confidence 100.0%)
   1H: Unknown (unavailable, confidence 99.3%)
   15m: Bull Trending (bullish, confidence 100.0%)
================================================================================
```

The 1D regime (Mean Reverting) is neutral, so nothing is confirmed (`primary_neutral`); the
1H regime is Unknown and counts as unavailable. See [docs/api.md](docs/api.md) for how
direction, agreement and confirmation are computed.

With live data a `💰 STATISTICAL ARBITRAGE:` block appears when there are mean-reversion or
momentum-breakdown signals. Confidence is frequently 100%: the models are overparameterized,
so treat it as uncalibrated (see [docs/status.md](docs/status.md)).

## CLI Commands

```bash
uv run mra current-analysis --symbol SPY
uv run mra detailed-analysis --symbol SPY --timeframe 1D
uv run mra generate-charts --symbol SPY --timeframe 1D --days 60 --output spy.png
uv run mra multi-symbol-analysis --symbols "SPY,QQQ,IWM"
uv run mra position-sizing --base-size 0.02 --regime "Bull Trending" --confidence 0.8
uv run mra export-csv --symbol SPY --filename analysis.csv
uv run mra continuous-monitoring --symbol SPY --interval 300   # --once / --max-iterations N
uv run mra regime-forecast --symbol SPY --steps 10
uv run mra calibrate-multipliers --symbol SPY --method sharpe_weighted
uv run mra backtest --symbol SPY --period 5y                   # walk-forward vs buy & hold
uv run mra list-providers
uv run mra start-api --dev                                     # binds 127.0.0.1 by default
uv run mra-optimize --mode grid --symbol SPY --provider yfinance
```

### Backtesting a parameter set

`mra backtest` evaluates one set of `RegimeStrategy` parameters and prints total return,
CAGR, Sharpe, Sortino, Calmar, max drawdown (and its duration), win rate, profit factor,
trades, time in market and average exposure next to buy-and-hold over the same bars.

- `--mode walk-forward` (default) refits the HMM on past data only and stitches the test
  windows into one **out-of-sample** curve, with a per-window table and the HMM refit count
  and failures. Tune the windows with `--train-bars`, `--test-bars` and `--retrain-every`.
- `--mode simple` fits the HMM once on the whole period. The output is labelled
  **IN-SAMPLE** because the model has seen every bar it trades.
- `--params file.json` takes a flat parameter object (`{"bull_mult": 1.5, "stop_loss": 0.03}`)
  or the file `mra-optimize` writes (its `best_params` are used). Unknown keys and bad
  values are rejected with a message. Without `--params` the defaults are used.
- `--json` prints a JSON object instead of the text report; `--output trades.csv` writes
  the trades. `--cost-model` picks `equity` (default), `retail`, `futures`, `hft` or `none`.

The HMM defaults (4 states, refit every 20 bars, 252/63-bar windows) match `mra-optimize`,
so the two can be chained:

```bash
uv run mra-optimize --mode random --symbol SPY --output best.json
uv run mra backtest --symbol SPY --params best.json --output trades.csv
```

When the parameters come from an `mra-optimize` file and test windows overlap the
optimizer's search period, the report is labelled `OUT-OF-SAMPLE (regime model only)` (JSON
`params_out_of_sample: false`) and warns how many windows overlap: they are out-of-sample for
the HMM but not for the parameters. Only the optimizer's holdout period is out-of-sample for
both.

In walk-forward mode buy-and-hold is restarted at each test window, like the strategy, so
both cover the same bars; the report also shows the asset's plain close-to-close return over
the whole span (`asset_return`).

`--provider` and `--api-key` work before or after the subcommand
(`mra --provider polygon current-analysis` or `mra current-analysis --provider polygon`).
The default provider is `yfinance`; set `DEFAULT_PROVIDER` to change it. API keys are only
checked by commands that fetch data, so `--help`, `list-providers`, and `position-sizing`
never need one. Commands exit non-zero when the analysis, chart, or export fails (for
`current-analysis`, when every timeframe fails); add `--debug` for a full traceback.

## Data Providers

| Provider | API Key | Setup |
|----------|---------|-------|
| Yahoo Finance (default) | Not required | `--provider yfinance` |
| Mock (offline) | Not required | `--provider mock` |
| Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | `--provider alphavantage` |
| Polygon.io | `POLYGON_API_KEY` | `--provider polygon` |
| Alpaca | `APCA_API_KEY_ID` + `APCA_API_SECRET_KEY` | `--provider alpaca` |
| Tiingo | `TIINGO_API_KEY` | `--provider tiingo` |

All providers return float OHLCV columns on a tz-naive index: intraday bars are labeled in UTC, daily bars by session date. Requests use `ProviderConfig.timeout`/`retries` and a client-side rate limiter shared by every instance with the same provider and key. They raise `InvalidSymbolError` (a `ValueError`), `AuthError` or `RateLimitError` (both `ConnectionError`s) so callers can tell a bad symbol from a network problem.

The mock provider generates a deterministic series per symbol, for demos, tests, and offline work.

Alpha Vantage's free tier returns **unadjusted** daily prices (splits show up as large one-day moves) and only about the last 30 days of intraday bars. Full daily history is also premium-only, so free keys get only the latest 100 daily bars. With a premium key, set `ALPHA_VANTAGE_PREMIUM=1` to get full, split/dividend-adjusted daily history. Weekly, monthly, and intraday bars are always adjusted. The free tier also allows only 25 requests per day.

Yahoo Finance serves sub-hourly bars for the last 60 days only, and hourly bars for the last 730 days. Requests outside these windows fail up front.

Alpaca uses the free IEX feed by default. Its volume covers only IEX trades, so set `ALPACA_DATA_FEED=sip` for consolidated volume. On the free plan, SIP data is delayed 15 minutes. To pass Alpaca keys with `--api-key`, use the form `KEY_ID:SECRET_KEY`. Alpaca intraday bars are limited to regular trading hours (09:30–16:00 ET) so they match the other providers. To keep pre- and post-market bars, pass `extended_hours=True` in the provider config.

## Mathematical Approach

### Hidden Markov Models

One model is used everywhere: `TrueHMMDetector` (`true_hmm_detector.py`), a Gaussian HMM (`hmmlearn`, Baum-Welch training) on six causal, stationary features with diagonal covariances and multiple EM restarts. States are labelled by absolute thresholds on their de-standardized means, and per-bar history uses filtered (forward-only) posteriors. The analyzer, `regime-forecast` and walk-forward validation therefore report the same model. See `packages/mra_lib/README.md` for details.

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
# Custom periods for different timeframes (defaults: mra_lib.config.timeframes.DEFAULT_PERIODS)
periods = {
    "1D": "2y",    # Daily data for 2 years
    "1H": "6mo",   # Hourly data for 6 months
    "15m": "1mo"   # 15-min data for 1 month
}

from mra_lib import MarketRegimeAnalyzer
analyzer = MarketRegimeAnalyzer("SPY", periods=periods)
```

## Dependencies

- **pandas**, **numpy** — Data manipulation and numerical computing
- **scikit-learn** — Gaussian Mixture Models
- **hmmlearn** — True HMM implementation (Viterbi decoding)
- **click** — CLI framework
- **yfinance**, **polygon-api-client**, **requests** (Alpha Vantage, Alpaca, Tiingo) — Market data providers
- **statsmodels** — Cointegration tests for pairs
- **matplotlib** — Visualization
- **fastapi**, **uvicorn**, **pydantic** — Web API
- **python-jose** — JWT auth; **uvicorn[standard]** provides the WebSocket implementation

Python 3.13+ required. All deps managed via `uv`.

## Testing

```bash
just test        # All tests
uv run pytest    # Or directly with pytest
```

Tests live next to each package (`packages/<pkg>/tests/`) and run offline with the `mock`
provider; tests that call live provider APIs are marked `integration` and excluded from
`just test-unit` / `just test-cov`. Coverage is enforced at 65% (currently about 92%). They
cover the analyzer and both detectors, providers and their contracts, risk sizing, the
backtest engine (including flat-price long/short round trips), metrics, walk-forward and the
optimizer, the calibrator, every CLI command, and the web API (auth matrix, error envelope,
rate limits, CSV/PNG responses, WebSocket lifecycle).

## Contributing

Contributions are welcome. Please ensure:

- Type hints throughout
- Google-style docstrings (`Args:` / `Returns:` / `Raises:`)
- Unit tests for new functionality
- `just qa` passes before submitting

## Development

```bash
just qa          # Format check + lint + type-check (run before committing; `just fix` to autofix)
just test        # All tests
just test-unit   # Unit tests only (no integration/slow)
just test-lib    # Core library tests only
```

### Docker

The image runs the web API on port 8000; compose publishes it on `127.0.0.1` only and requires a
`JWT_SECRET` of at least 32 characters:

```bash
cp .env.example .env   # set JWT_SECRET (and any provider keys)
just docker-up         # docker compose up -d
just docker-health     # GET http://127.0.0.1:8000/health
```

The image also contains the CLI and the token minter:

```bash
docker run --rm --entrypoint mra market-regime-analysis --provider mock current-analysis
docker compose exec api mra-token --sub alice   # uses the container's JWT_SECRET
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
