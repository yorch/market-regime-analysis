# Code Review — 2026-10-03

Exhaustive multi-agent review of `main` @ `388235a` (line numbers refer to that commit; #10 and #11 landed afterwards and added `credentials.py`, which resolves the duplicated CLI/web API-key lookup). Eight parallel reviewers covered: analytics
correctness, backtesting correctness, security, web API, data providers + CLI, code quality /
architecture, tests / CI / tooling, and docs / features. Findings were verified by running code
(mock/synthetic data, `TestClient`, `CliRunner`) unless marked *(plausible)*. Duplicates reported by
several reviewers are merged; the reviewer count is shown as `×N` where it adds confidence.

## Baseline

| Gate | Result |
|---|---|
| `pytest` | 472 passed, 1 failed (`test_system.py` hits live Yahoo), ~27s |
| Coverage | 68.7% reported → **~67% real** (the `"pass"` exclude regex drops every `@click.pass_context` command) |
| `ruff check` / `ruff format --check` | clean (aided by broad per-file ignores) |
| `mypy packages/` | **16 errors / 9 files** (CI typecheck is soft-fail); 145 under `--strict` |
| `uv lock --check`, `uv build` | clean |
| `docker compose up` | **crash-loops** (`unrecognized arguments: --no-dashboard`) |

## Remediation status (updated after #11–#27, `main` @ `10236d4`)

Every item below was re-checked against `10236d4` by reading code and tests, not commit
messages. Items not listed under **Partial** or **Open** are done.

| Gate | At review (`388235a`) | Now (`10236d4`) |
|---|---|---|
| `pytest` (non-integration) | 472 passed, 1 failed | **959 passed** |
| Coverage (anchored exclude regex) | ~67% | **94%** |
| `mypy packages/` | 16 errors, soft-fail | **0 errors, blocking in CI** |
| `print()` in `mra_lib` | 157 | **0** (ruff T201 enforced) |
| `docker compose up` | crash-loops | boots; CI smoke-tests `mra --help` and `/health` |

| PR | Resolved |
|---|---|
| #11 `feat(providers)` | Alpaca + Tiingo providers; shared `credentials.py` (removes the duplicated CLI/web key lookup); retrying `_http.get_json` |
| #13 `ci` | Dependabot for uv and GitHub Actions (first bump: #21) |
| #14 `fix(cli)` | `generate-charts` repair after #17/#19; mypy made blocking |
| #15 `fix(deploy,ci)` | P0-16 (compose/Docker), coverage regex, integration markers, CI permissions/secret scoping/`--locked`/SHA pins, pre-commit local hooks, dependency declarations, non-root volume ownership, `.env.example` rewrite |
| #16 `fix(analytics)` | P0-11 (persistence/transition), P0-12 (GMM feature lookup), risk sizing (no-edge floor, hedge headroom, vol targeting, Kelly cap), multiplier saturation, timeframe annualization, returns-based correlations + Engle-Granger pairs, shared `config/regime_tables.py`, scoped warnings |
| #17 `fix(web)` security | P0-1…6 (token minting removed → `mra-token`, strong `JWT_SECRET`, env `API_KEYS` via `X-API-Key`, CSV built in memory, generic errors + log scrubbing, authenticated WebSocket), production default, CORS, path-based rate limiter, input caps, P0-7 headless charts |
| #18 `fix(backtesting)` | P0-10 (short accounting), Sharpe/Sortino/Calmar, stitched drawdown, gap-aware stops, half-spread + market impact, unified trade stats, regime cache (~20× faster optimizer), holdout split, seeded search |
| #19 `fix(cli,providers)` | P0-13/14/15 (15m period, yfinance default, lazy keys, per-subcommand `--provider`), `mock` provider, exit codes, resilient monitoring, provider timeouts/error types/rate limiter, AV adjusted + period trim, Polygon pagination, tz contract, runnable examples |
| #20 `fix(web)` stability | P0-8/9 (WebSocket lifecycle + event loop), thread-pool timeout + concurrency cap, single error envelope, strict JSON, per-timeframe loading, multi-symbol via portfolio APIs |
| #21 `ci(deps)` | GitHub Actions bumps (Dependabot) |
| #22 `chore` | `mra start-api --dev` sets `ENVIRONMENT=development` (shared `serve()`); `slowapi` + `alpha-vantage` removed; `mra --help` smoke check; docs rewritten against the code (`docs/api.md`, `docs/status.md`, README, AGENTS.md, `.env.example`, copilot instructions) |
| #23 `docs` | Remaining doc inaccuracies after #22 |
| #24 `fix(web)` | Multi-symbol all-failed requests classified by root cause; pre-commit mypy comment |
| #25 `refactor(lib)` | `print` → `logging` (NullHandler), `format_*()` reports with deprecated `print_*` shims, `mra_lib/errors.py` hierarchy, provider errors no longer re-wrapped as `ValueError`, per-symbol portfolio failures exposed, `types/protocols.py` deleted, ruff `G`/`BLE` enabled, detector raises `ModelNotFittedError`/`InsufficientDataError` |
| #26 `refactor(indicators)` | One detector (`TrueHMMDetector`) for analyzer, `regime-forecast`, walk-forward and optimizer (GMM is a deprecated alias); `RegimeDetector` protocol + injectable `detector_factory`; shared causal, scale-free 6-feature set (`indicators/features.py`); diag covariance, `min_covar`, EM restarts, parameter-scaled sample minimum, BIC `select_n_states`; absolute-threshold regime mapping with volatility-sorted state order; filtered (forward-only) posteriors; per-bar causal volume handling |
| #27 `chore(web)` | Config/API cleanup: `RATE_LIMIT_BURST`, position-sizing response, compose env forwarding, duplicate `/health`/`/metrics` routes, single `__version__` from package metadata |

### Partial

- **Model confidence** — less saturated after #26 (5th percentile 1.000 → 0.74–0.96), but the
  median is still ≈ 0.999; serially correlated rolling features are treated as independent
  evidence. Needs calibration (tempering or less overlapping features).
- **Optimizer** — holdout split exists (`holdout_frac`, CLI default 0.2), but no deflated Sharpe /
  multiple-testing correction.
- **Calibrator** — optional holdout and `in_sample` flag, but default `holdout_frac=0.0`,
  `min_trades_per_regime=5`, PnL still attributed to the entry regime.
- **Alpha Vantage adjustment** — free-tier daily is still unadjusted (documented, one-time warning).
- **In-progress bar** — dropping it is opt-in (`drop_incomplete_bar=True`).
- **Web settings** — hand-rolled `os.getenv` parsing, not `pydantic-settings`.
- **Model caching** — the web API builds and fits a new analyzer per request; only the backtest
  regime cache exists.
- **CORS** — an explicit `CORS_ORIGINS=*` is accepted (warning only) and also allows any WebSocket
  Origin (`mra_web/config.py:171`, `websocket.py:170`).
- **Typed results** — dataclasses in backtesting; analyzer, backtest result and trade payloads are
  still dicts.
- **Blind excepts** — 36 `except Exception` remain in `src`, 15 annotated with a `noqa: BLE001`
  reason.
- **Charts** — `plot_regime_analysis` returns a pyplot-registered figure; callers that ignore it
  leak it (no production callers). matplotlib is still a hard dependency of `mra_lib`.
- **Test fixtures** — shared synthetic OHLCV fixture in `conftest.py`, but 6 test files still define
  their own builder.

### Open

**Architecture / code quality**
- Analyzer does network I/O and model fitting in `__init__` (`analyzer.py:112-114`).
- `continuous-monitoring` reloads and refits every tick (`analyzer.py:856-858`).
- `ProviderConfig` dynamic `setattr` (`data_providers/base.py:130`).
- `analyzer.py` per-file ruff ignores still include `B904` and complexity rules.

**Decision needed**
- Backtest `strategy.py` keeps its own regime→strategy map (BEAR→TREND_FOLLOWING,
  HIGH_VOL→DEFENSIVE) instead of the shared `REGIME_STRATEGIES`.

**Small**
- Rate limits, metrics and WebSocket caps are in-memory per worker process.
- `.claude/` and `.vscode/` are tracked; `.claude/settings.json` allows `Bash(rm:*)` and `Bash(env)`.
- `python-jose` → consider PyJWT; run `uv audit`/`pip-audit` online against the lockfile.
- Alpaca/Tiingo are verified only with mocked HTTP; needs a live-key smoke run.

**Missing tests**
- #3 walk-forward windows never overlap and refits use only training data. (#2 is covered for
  features and filtered posteriors by `test_features.py::test_truncation_invariance` and
  `test_true_hmm.py::test_filtered_posteriors_are_causal`.)

**Features** (table at the end of this document)
- Done: #1 (basics), #10 (headless PNG charts, in-memory CSV).
- Partial: #2 (single detector + BIC state selection; no model persistence), #7 (regime cache +
  holdout + seed; no parallelism or deflated Sharpe), #9 (filtered posteriors exposed; no
  per-state explanation in the API), #11 (token bucket, retries, timeouts, tz contract; no disk
  cache or `start`/`end`), #12 (Engle-Granger; no half-life), #14 (Tiingo, Alpaca, mock; no
  CCXT, Stooq, FRED or CSV/Parquet).
- Not started: #3, #4, #5, #6, #8, #13.

---

## P0 — Critical

### Security (web API)
1. **Anyone can mint a JWT** — `POST /auth/token?username=admin` issues a 24h token with no credential
   check, in production too. `app.py:180-199`. A test (`test_web_smoke.py:515`) locks this in. ×4
2. **Forgeable tokens** — default secret is the public string `your-secret-key-change-in-production`
   (`config.py:59`); compose passes `JWT_SECRET=""` when unset → HS256 with empty key.
   `.env.example` falsely says "auto-generated if empty". No `exp` required on decode
   (`auth.py:70-91`). ×4
3. **Hard-coded API keys** `demo-api-key-12345` / `admin-api-key-67890` always valid
   (`auth.py:161-164`), read from a `?api_key=` query param, non-constant-time compare. ×3
4. **Arbitrary file write** — `/api/v1/export/csv` passes client `filename` to `to_csv()`;
   `../../...` verified to escape. Response also fabricates `records_count=100` and path.
   `endpoints.py:546-565` → `analyzer.py:674-735`. ×4
5. **Provider API key leaks to clients** — alpha_vantage puts `apikey=` in the URL; on transport
   errors the URL lands in `str(e)`, which endpoints return as `detail` and the WS sends as
   `{"error": ...}`. `endpoints.py:95-111,183-188,512-518,576-581`, `websocket.py:254-263`.
6. **Unauthenticated WebSocket** `/ws/monitoring/{symbol}` (+ `/ws/test`, `/ws/monitoring/status`),
   no Origin check, burns server provider quota. `websocket.py:109-170,289-310`.

### Web API stability
7. **One request kills the server** — `/charts/generate` calls pyplot `plt.show()` in a worker thread:
   SIGABRT on macOS; leaks a figure per call on Agg; returns a fake file path.
   `endpoints.py:482-491`, `analyzer.py:564-631`. ×3
8. **WebSocket zombie loops** — `send_personal_message` swallows send errors and the loop never
   reads the socket, so after disconnect it keeps fetching + refitting forever (954 iterations in
   50 ms in a stubbed test). `websocket.py:68-74,191-279`.
9. **WebSocket blocks the event loop** — analyzer construction (network + 3 HMM fits) runs inline in
   `async def`. Every tick freezes all HTTP/WS traffic. `websocket.py:194-199`. ×2

### Analytics / backtesting correctness
10. **Shorts lose their entire notional** — opening a short only deducts costs, never credits proceeds;
    close deducts `price*shares`. Flat-price, zero-cost 20% short ends at −20% while trade PnL = 0.
    Every result with `bear_short=1` is wrong and biases the optimizer. `engine.py:264-267,320-323,403-405`.
11. **Persistence always 0, transition prob always 0.5, risk level always "High"** — windows are
    ≤20 bars but gated on `>= 50`, so `recent_predictions` is always empty. Affects every CLI/API/
    portfolio output. `analyzer.py:444-459`. ×2
12. **GMM detector reads wrong feature columns** — `X[:, 9]` "trend_strength" is actually `sma_9`
    (standardized price level); thresholds 0.001/0.002 applied to z-scores. BULL/BEAR/MR labels are
    driven by price level. `hmm_detector.py:248-256`. ×2

### CLI / deployment
13. **Every analyzer command fails on yfinance** — default 15m period `"2mo"` isn't supported by the
    yfinance provider (`analyzer.py:49-53`, `main.py:546`). README quickstart fails.
14. **CLI unusable without an Alpha Vantage key** — group default `--provider alphavantage` with eager
    key validation; `list-providers`, `position-sizing`, even `<cmd> --help` fail.
    `main.py:138-166`. ×2
15. **Docs put `--provider` after the subcommand** → `No such option`. `README.md:29,116`,
    `AGENTS.md:33,51-52`, `packages/mra_cli/README.md:14`. ×2
16. **Docker compose stack can't start** — `command: ["--no-dashboard"]` (and `just docker-once`
    `--once`) rejected by `mra-api`; ports 8080 vs 8000; ~25 env vars (`SCAN_*`, `TELEGRAM_*`,
    `DISCORD_*`, `EMAIL_*`, `DATABASE_URL`, `WEB_*`, `DEFAULT_*`, …) are read by nothing — the
    "scanner-first architecture" from #6 exists only in config. Postgres override is unused. ×4

---

## P1 — High

### Analytics
- **Non-stationary features** (raw `sma_*`, `atr`, `price_change`) in both detectors → states
  cluster by price era. `hmm_detector.py:76-100`, `true_hmm_detector.py:108-119`.
- **TrueHMM regime mapping is structurally biased** — 75th/25th percentile of 6 state means always
  labels 2 HIGH_VOL + 2 LOW_VOL; labels change with seed; MEAN_REVERTING is a catch-all.
  `true_hmm_detector.py:317-329,453-462`.
- **Overparameterized models → confidence ≈ 1.0 always** (22 collinear features, full covariance,
  ≥30 rows; cond ≈1e6; hmmlearn log-likelihood decreasing yet `converged=True`).
- **Positions taken with negative edge** — `max(0.01, …)` floors; Kelly=0 still returns 1%.
  `risk_calculator.py:397,444,494,563-573`.
- **Exposure clamp blocks hedges** — net headroom ignores trade direction.
  `risk_calculator.py:204-205`.
- **Portfolio correlations on price levels** (spurious; 60% of independent random walks |ρ|≥0.3).
  `portfolio.py:122,203,254`, `endpoints.py:265`. `docs/status.md:41` falsely claims this is fixed. ×4
- **Analyzer uses the GMM detector; walk-forward validates TrueHMM** — validation numbers say
  nothing about the model users see; `regime-forecast` and `current-analysis` can disagree. ×3

### Backtesting
- **Optimizer reports in-sample as out-of-sample** — params selected on the same walk-forward
  windows, "validation" re-runs on the same data; 1,728 trials, no holdout / deflated Sharpe.
  `optimizer.py:139-262`, `mra_cli/optimization.py:233-264`.
- **Calibrator fits and evaluates on the same data**, `min_trades_per_regime=5`, PnL attributed to
  entry regime only. `calibrator.py:311-381`.
- **HMM recomputed per parameter set** — ~87% of runtime redundant; CLI grid ≈ 5.4 h on 5y data;
  `predict_regime` is O(n²) over growing prefixes. `optimizer.py:103-119`, `walk_forward.py:73-125`.

### Web API
- **Missing-key error → 500 with empty body** (`datetime` in `HTTPException.detail`).
  `utils.py:108-118`, `app.py:120-132`.
- **`HTTPException` swallowed into 500** with Python-repr detail. `endpoints.py:95-111,512-518,576-581`.
- **`HTTPBearer()` auto_error** → 403 before dev bypass / API-key path; all `docs/api.md` curl
  examples fail. `auth.py:22`. ×3
- **Dev mode accepts any bearer and is the default** (Dockerfile sets no `ENVIRONMENT`); `/debug/config`
  exposed. `config.py:44`, `auth.py:187`.
- **No rate limits on `/api/v1/*`**; unbounded `symbols` list; no provider timeouts; `config.timeout`
  unused. `app.py:34,96`, `models.py:85-93`. ×3
- **Analyzer always loads all 3 timeframes** → 3× cost; any one TF failure fails 1D-only requests.
  `endpoints.py:75,483,546`, `analyzer.py:49-86`. ×3
- **NaN/Infinity emitted as invalid JSON**; `np.bool_` unserializable. `utils.py:43-47`.

### Providers
- **Alpha Vantage returns unadjusted prices** (splits look like −50% returns) while others are
  adjusted. `alphavantage_provider.py:121`.
- **Alpha Vantage ignores `period`** (2y → 20y history). `alphavantage_provider.py:111-112`.
- **No request timeouts / retries / rate limiting** — `ProviderConfig.timeout/retries/rate_limit`
  are never read; Polygon advertises 60/min but comment says 5. `base.py:18-20`.

### Tests / CI
- **Coverage inflated** by `exclude_lines = ["pass"]` (unanchored regex). `main.py` real: 53%.
- **`integration-test` CI job always fails** (no test has the marker → pytest exit 5).
- **Live-network `test_system.py` runs in the unit job** (flaky in CI, asserts nothing).
- `test_backtest.py` / `test_true_hmm.py` contain **zero tests** (scripts named `test_*`).

### Code quality
- **`mra_lib` is not UI-free** — 157 `print()`s, `print_*`/plot/monitor/export methods in the lib,
  matplotlib as hard dep, zero `logging`. ×2
- **`types/protocols.py` is dead** — nothing implements or references the protocols (0% coverage);
  CLAUDE.md claims otherwise. ×3

---

## P2 — Medium

**Analytics / risk**: vol targeting double-counts (`target*hist/cur²`, `risk_calculator.py:482`);
Kelly cap bypass at `win_rate==1` (`:319`); invalid inputs silently return a position (`:579`);
position multiplier saturates at 0.5 so Bull == Bear (`analyzer.py:262-269`); `sqrt(252)`
annualization for intraday (`analyzer.py:174`, `metrics.py:81,101`); GMM transition rows can be all
zero (`hmm_detector.py:217`); label flips bar-to-bar and HMM state ints aren't stable across refits;
`predict_proba` is smoothed but documented as filtered (`true_hmm_detector.py:243`); module-level
`warnings.filterwarnings("ignore")` in 6 modules; `np.random.seed(42)` in mock provider.

**Backtesting**: walk-forward max DD is per-window (−13.5% reported vs −25.6% stitched,
`walk_forward.py:285`); Calmar positive for losers (`metrics.py:242`); Sortino formula wrong /
explodes (`metrics.py:104`); stop fills ignore gaps (`engine.py:364`); profit factor 999 vs 0
sentinel across three duplicate implementations; `random_search` unseeded; failed refit leaves
unfitted model → silent UNKNOWN.

**Web**: five different error envelopes, no `RequestValidationError` handler; multi-symbol blank
symbols → 503, bogus `correlation_risk=-1`, each symbol analyzed 3×; WS `interval=abc` crashes,
provider unvalidated, wrong close codes; `run_in_thread` uses default executor, no timeout, rejects
kwargs; no caching of fitted models (docs claim caching); unbounded `response_times` list;
hand-rolled settings instead of `pydantic-settings` (`LOG_LEVEL=info` crashes); CORS `*` + credentials
reflects any origin; unauthenticated `/metrics`, docs, `/health` with environment.

**Providers / CLI**: all exceptions wrapped as `ConnectionError` ("Network error" for bad symbol);
inconsistent tz (yfinance NY-aware, Polygon naive UTC, AV naive ET) and extended hours;
in-progress bar not dropped; AV `30m`/`1w` mapping mismatch; Polygon `quarter`/`year` advertised but
raise, `re.match` vs `fullmatch`, non-paginating `get_aggs` *(plausible truncation)*;
`continuous-monitoring` exits 0 on first transient error, drifts, retrains every tick; lib swallows
errors so `detailed-analysis`/`generate-charts`/`export-csv` exit 0 on failure; `generate-charts`
writes nothing and `--days` means bars; mock provider unregistered and CLI `--provider` is a hard-coded
`Choice` (no offline mode); `examples/*.py` fail as documented (`uv run examples/x.py` →
`ModuleNotFoundError`), notebook imports old `market_regime_analysis` package; `mra-optimize
--provider polygon|alphavantage` never reads the key.

**DRY / architecture**: feature engineering copied 3× (both detectors + analyzer); TrueHMM's two
`_map_state_*` methods are copies; regime multiplier table ×4 with conflicting UNKNOWN (0.0 vs 0.2);
regime→strategy maps conflict (`analyzer.py:239` vs `strategy.py:111`); `validate_api_key` + env map
duplicated CLI/web; timeframe list literal ×9; version `"1.0.0"` ×7; duplicate `/health`,
`/metrics`; ~25 lines of boilerplate per endpoint; analyzer god-class does I/O in `__init__`;
37 blind `except Exception`; stringly-typed dicts for results/trades/directions.

**Tooling**: mypy soft-fail with real Optional bugs (`true_hmm_detector.py:237,518,523,552,561`);
pre-commit mypy v1.11 isolated env vs locked 1.19; `mra-cli` imports `uvicorn`/`mra_web` undeclared;
unused deps (`statsmodels`, `requests`, `websockets`, `python-multipart`); named volumes root-owned
vs uid 1000; ruff per-file ignores hide T201/B904/complexity.

## P3 — Low / hygiene

CI: no top-level `permissions`, provider secrets exposed to every job, `uv sync` without `--locked`,
tag-pinned (not SHA) actions, `uv:latest` image; Dockerfile lacks `HEALTHCHECK`/`EXPOSE`;
`just qa` rewrites files (not a check); `[tool.uv] dev-dependencies` deprecated; no `conftest.py`
(OHLCV builder ×5); no `--strict-markers`/`filterwarnings`; Pydantic v1 `.dict()/.json()` (12×);
`datetime.utcnow()`; `example_new_provider.py` shipped in package; `ProviderConfig` dynamic setattr;
docstrings are Google style though CLAUDE.md says numpy; `.claude/` and `.vscode/` tracked despite
`.gitignore`; committed `.claude/settings.json` allows `Bash(rm:*)` and `Bash(env)`; dependency floors
(`python-jose>=3.3.0`) allow vulnerable versions without the lock — consider PyJWT; run
`uv audit`/`pip-audit` online to confirm lock advisories *(plausible: urllib3, aiohttp,
python-multipart, starlette)*.

## Documentation corrections

- README/AGENTS/mra_cli README: `--provider` placement; "no API key needed" claim; example output
  (position size line is actually the multiplier); missing `continuous-monitoring`.
- `.env.example`: lists ~25 unused vars, omits all real `API_*`/`JWT_*`/`CORS_*`/`RATE_LIMIT_*` vars,
  wrong JWT claim, `DEFAULT_PROVIDER` unused.
- `docs/api.md`: wrong auth description (`X-API-Key` "future" vs hard-coded query keys), wrong
  `metrics` keys, wrong error body, false claims (caching, parallel timeframes, rate limits),
  nonexistent `examples_api` import, Docker snippet differs from real Dockerfile, undocumented routes.
- `docs/status.md`: stale (8 vs 11 commands, test counts, CI coverage already enforced), false
  "correlation fixed", unreproducible performance numbers.
- `AGENTS.md`: protocols "implemented" claim, nonexistent `config/settings`, "Four jobs" lists five
  (six exist), docstring style.
- `.github/copilot-instructions.md`: references `PLAN.md`, `examples.py`, TA-Lib.
- "Jim Simons / Renaissance methodology" wording in CLI/API help contradicts README disclaimer.

## Most valuable missing tests

1. Flat-price long/short round trips at zero cost end at initial capital.
2. Look-ahead / truncation invariance: changing future bars leaves past regimes and signals unchanged.
3. Walk-forward windows never overlap; detector refit only on train.
4. Metric formulas vs hand-computed values (Sharpe, Sortino, Calmar sign, DD + trailing duration).
5. Analyzer persistence > 0 and risk level not constant.
6. Web auth matrix (no token / tampered / expired / valid) and `/auth/token` must not mint freely.
7. `/api/v1/*` happy + error paths via `TestClient` with a registered mock provider; NaN serialization.
8. WebSocket connect → update → disconnect → loop task terminates.
9. CLI analysis commands via `CliRunner` + mock provider; `mra-optimize` tiny grid.
10. Docker smoke: build, run, poll `/health`.

## Feature proposals (prioritized)

| # | Feature | Effort |
|---|---|---|
| 1 | Make basics work: default yfinance, per-subcommand `--provider`, lazy key check, register `mock` provider, fix compose | S |
| 2 | Single detector (TrueHMM, stationary features, BIC state selection) + model persistence | M |
| 3 | Regime history store (`DataStoreProtocol` → SQLite/Postgres) + `/regimes/{symbol}/history` | M |
| 4 | Scheduled scanner + regime-change alerts (Telegram/Discord/webhook/email) — what compose advertises | M |
| 5 | Load optimized/calibrated params (`--params file.json`), single `RegimeConfig` | S |
| 6 | `mra backtest` CLI + async backtest endpoint with buy-and-hold benchmark | S–M |
| 7 | Signal cache in optimizer (regimes once per window) + joblib parallelism; holdout / deflated Sharpe | M |
| 8 | Multi-timeframe confirmation signal | S–M |
| 9 | Explainability: per-state feature z-scores, state probabilities, label rationale | M |
| 10 | Headless charts (Agg → PNG bytes), streaming CSV export | S |
| 11 | Provider infra: on-disk cache, token-bucket limiter, retries, tz normalization, `start/end` | M |
| 12 | Cointegration pairs (Engle-Granger, half-life) | M |
| 13 | Regime-conditioned portfolio allocation; paper-trading loop with DD kill-switch | L |
| 14 | New providers: CCXT/crypto, Tiingo, Stooq, FRED covariates, CSV/Parquet | S each |

## Suggested fix plan (one PR each)

1. **security/web-auth** — remove open token minting, require strong `JWT_SECRET`, drop hard-coded
   keys (`APIKeyHeader` + env, `compare_digest`), `auto_error=False`, default `production`, require
   `exp`, auth + Origin on WS, rate-limit defaults, symbol regex + list caps, generic error details.
2. **fix(web)** — streaming CSV export (no server path), Agg/BytesIO charts, WS disconnect + threadpool,
   error-envelope handlers, `model_dump(mode="json")`, NaN-safe JSON, per-timeframe analyzer loading.
3. **fix(backtesting)** — short accounting, Calmar/Sortino, stitched DD, gap stops, unified trade stats
   + regression tests.
4. **fix(analytics)** — persistence/transition windowing, GMM feature lookup by name, risk floor /
   hedge headroom / vol targeting, returns-based correlations.
5. **fix(cli/providers)** — `1mo` 15m period, default yfinance + lazy key, register mock, timeouts,
   AV adjusted + period trimming, exit codes.
6. **fix(deploy/ci)** — compose command/ports/env, volume ownership, coverage regex, integration
   markers, CI permissions/secrets scoping, `uv sync --locked`, pre-commit local hooks.
7. **docs** — README, `.env.example`, `docs/api.md`, `docs/status.md`, AGENTS.md.
8. **refactor** — shared features module + single detector, regime tables, logging instead of print,
   typed results, protocols wired or removed, tighten mypy/ruff.
