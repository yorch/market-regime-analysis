# Project Status — Market Regime Analysis

**Date**: 2026-10-03

## Summary

A research tool that classifies market regimes with Hidden Markov Models and derives
regime-aware signals and position sizes, exposed as a library (`mra_lib`), a CLI (`mra`,
`mra-optimize`) and a REST/WebSocket API (`mra-api`). After the October 2026 code review
(issue #12) and fix PRs #13–#20, the code paths users touch work end to end, are tested, and
type-check cleanly. **The trading strategy is not validated: it has not been shown to
outperform buy-and-hold.** Treat every output as research, not advice.

## Quality Gates

| Gate | At review (`388235a`) | Now |
|------|-----------------------|-----|
| `pytest -m "not integration"` | 472 passed, 1 failed | 882 passed |
| Coverage (`just test-cov`; CI fails under 65%) | ~67% (inflated by an unanchored `pass` exclusion) | 92% |
| `mypy packages/` | 16 errors, soft-fail in CI | 0 errors, **blocking** in CI |
| `docker compose up` | crash-looped | boots; CI smoke-tests `/health` and `mra --help` in the image |

CI (`.github/workflows/ci.yml`) runs lint, format check, mypy, unit tests with coverage and
package builds on every PR, then builds and smoke-tests the Docker image. Integration tests
(live provider APIs) run only on manual dispatch. Dependabot opens weekly PRs for uv packages
and GitHub Actions (pinned to SHAs).

## What Was Fixed (October 2026)

| PR | Area | Main fixes |
|----|------|------------|
| #15 | deploy, CI | Runnable compose stack and Dockerfile (non-root, `HEALTHCHECK`), honest coverage regex, integration markers, CI least-privilege permissions, secrets scoped to the integration job, `uv sync --locked`, SHA-pinned actions, `.env.example` rewritten to the variables code reads |
| #16 | analytics | Regime persistence / transition probability windows (were always 0 / 0.5), GMM detector reads features by name (was reading price level), risk sizing (no-edge floor, hedge headroom, vol targeting, Kelly cap), timeframe-aware annualization, returns-based correlations, Engle-Granger pairs, shared regime tables |
| #17 | web security | No token-minting endpoint (`mra-token` CLI instead), strong `JWT_SECRET` required, `API_KEYS` via `X-API-Key`, CSV returned in the response (no server-side file write), generic error messages + log scrubbing, authenticated WebSocket with Origin check, production by default, CORS, path-based rate limiting, input caps, headless charts |
| #18 | backtesting | Short-sale accounting, Sharpe/Sortino/Calmar formulas, stitched walk-forward drawdown, gap-aware stops, half-spread + market impact costs, unified trade stats, per-window regime cache (optimizer ~20× faster), holdout split, seeded random search |
| #19 | CLI, providers | Working defaults (yfinance, valid 15m period), lazy API-key checks, `--provider` before or after the subcommand, offline `mock` provider, non-zero exit codes, resilient `continuous-monitoring`, provider timeouts / retries / typed errors / shared rate limiter, Alpha Vantage adjusted + period trimming, Polygon pagination, timezone contract, runnable examples |
| #20 | web stability | WebSocket lifecycle (no zombie loops) and event-loop safety, thread-pool timeout and concurrency cap, one error envelope, strict JSON, per-timeframe loading, multi-symbol analysis via the portfolio APIs |
| #13, #14 | tooling | Dependabot; mypy made blocking; `generate-charts` repaired |

Follow-up PR (`chore: post-fix follow-ups and documentation refresh`): `mra start-api --dev` now sets `ENVIRONMENT=development` (it
shares `mra-api`'s entry point), unused `slowapi` and `alpha-vantage` dependencies removed,
CLI smoke test in the Docker job, documentation refreshed.

## Known Limitations

- **No demonstrated edge.** Backtests have not beaten buy-and-hold. Optimizer results are
  in-sample with respect to parameter selection; a holdout split exists
  (`--holdout-frac`, default 0.2), but there is no multiple-testing correction (e.g.
  deflated Sharpe) for the number of trials.
- **Model fit quality.** One detector (`TrueHMMDetector`) is now used everywhere, on a
  stationary 6-feature set. Confidence is less saturated but still high in persistent
  states, and different seeds can reach different EM optima; treat confidence as
  uncalibrated.
- **Strategy maps differ.** The backtest `RegimeStrategy` regime→direction map differs from
  the shared `REGIME_STRATEGIES` table used for recommendations.
- **Calibrator attribution.** `calibrate-multipliers` attributes each trade's P&L to its
  entry regime only, not bar by bar.
- **Web API scale.** Rate limits, metrics and WebSocket caps are in process memory (per
  worker); fitted models are not cached.
- **Library hygiene.** `mra_lib` logs instead of printing and raises typed errors
  (`mra_lib/errors.py`), but still returns many results (backtests, trades) as dicts.
- **Data.** Alpha Vantage free tier: unadjusted daily prices, latest 100 bars, 25 requests/day.
  Yahoo Finance: 15m bars for the last 60 days only. No on-disk cache.

## Open Work

In progress:

- **Typed results** — backtest results and trades are still returned as dicts
  (detector unification #26 and logging/typed errors #25 have landed).

Needs a decision:

- Which regime→strategy map is canonical (backtest vs. shared table).

Next:

- Deflated Sharpe / multiple-testing correction in the optimizer.
- Bar-by-bar regime attribution in the calibrator.

Feature roadmap (from the review's proposals):

- Landed: regime history store and `GET /api/v1/regimes/{symbol}/history` (#29),
  multi-timeframe confirmation signal (#28), `mra backtest` vs buy-and-hold (#30), scheduled
  scanner with regime-change alerts (`mra scan`, `mra_lib.scanner`, compose `scanner`
  profile, #32). Scanner follow-ups: email/Slack channels, history retention/pruning.
- Model persistence and loading optimized/calibrated parameters into the analysis commands
  (`mra backtest --params` already reads `mra-optimize` output).
- Backtest API endpoint (can reuse `mra_lib.backtesting.run_backtest`).
- Explainability (per-state feature z-scores).
- Provider infrastructure: on-disk cache, `start`/`end` ranges; more providers (crypto, FRED).
- Regime-conditioned allocation and a paper-trading loop with a drawdown kill switch.

### Ideas backlog

Unimplemented ideas carried over from the January 2026 planning documents (the original
PR #2 branch), roughly in priority order. The strategy has no demonstrated edge, so the
first two items come first.

1. **Alternative strategies.** Only `RegimeStrategy` exists; add a small strategy interface
   so these can be backtested side by side with `mra backtest`:
   - *Regime as a risk overlay on buy-and-hold* — stay long, cut exposure in High Volatility
     (and raise it in Low Volatility). The simplest candidate and the most likely to help
     risk-adjusted returns.
   - *Regime-transition trading* — act on regime changes rather than regimes.
   - *Mean reversion within a regime* — enter on price z-score < −2 in Mean Reverting, exit
     at z = 0 or on a regime change.
   - *Volatility breakout* — Low → High Volatility transition plus a price breakout.
   - *Multi-timeframe alignment* — trade only when `confirm_timeframes` (#28) agrees.
   - *Pairs with a regime filter* — trade cointegrated pairs only when their regimes align.
2. **Robustness testing.** Monte Carlo / bootstrap confidence intervals on window returns,
   parameter-sensitivity sweeps, and in-sample vs out-of-sample degradation tracking
   (complements the deflated Sharpe item above).
3. **Multi-symbol validation.** Run the walk-forward backtest over a universe (SPY, QQQ, IWM,
   DIA; XLF, XLE, XLK, XLV; TLT, IEF; GLD, USO; EFA, EEM) and report per-symbol and
   aggregate results.
4. **Risk controls.** Drawdown-scaled sizing (e.g. −5% → size ×0.9, −10% → ×0.75,
   −15% → ×0.5) and a max-drawdown kill switch; VaR / CVaR; a correlation-with-open-positions
   limit in `PortfolioPositionLimits`.
5. **Data quality.** OHLC sanity checks for every provider (only Polygon has them), gap and
   stale-bar (zero-volume) detection, return/volume outlier flags, split detection,
   cross-provider comparison, and a per-dataset quality score.
6. **Model persistence and versioning** (save/load a fitted detector with its scaler, state
   map, training range and library version).
7. **Pairs trading depth.** Half-life / Ornstein-Uhlenbeck filter, Johansen test for
   baskets, Kalman-filter dynamic hedge ratios.
8. **Hurst exponent feature** — a cheap, scale-free trending vs mean-reverting signal that fits
   the #26 feature set.
9. **Trading operations** (once paper trading exists): P&L tracking, trade log / audit trail,
   large-loss and drawdown alerts, dashboard.
10. **Write-ups.** Methodology document and per-strategy tearsheets (`mra backtest` output
    covers part of a tearsheet).

## Historical Notes

Earlier assessments (March 2026) reported backtest figures such as "+9.50% total return,
−57.09% excess return vs buy-and-hold, Sharpe 0.18". Those numbers were produced before the
October 2026 fixes to short accounting, metric formulas and walk-forward evaluation, and are
not reproducible with the current code. They are kept here only as history; rerun
`uv run mra-optimize` for current figures. Older planning documents are in
[archive/](archive/).
