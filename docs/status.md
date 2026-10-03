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
- **Two detectors.** The analyzer (CLI/API) uses the GMM-based detector
  (`indicators/hmm_detector.py`); walk-forward validation and `regime-forecast` use the
  hmmlearn detector (`indicators/true_hmm_detector.py`). Validation results therefore say
  little about the model users see, and the two can disagree.
- **Model fit quality.** Features include non-stationary series; models are
  overparameterized, so reported confidence is often ≈ 1.0. The hmmlearn detector's
  percentile-based state labelling is biased. Treat confidence as uncalibrated.
- **Strategy maps differ.** The backtest `RegimeStrategy` regime→direction map differs from
  the shared `REGIME_STRATEGIES` table used for recommendations.
- **Calibrator attribution.** `calibrate-multipliers` attributes each trade's P&L to its
  entry regime only, not bar by bar.
- **Web API scale.** Rate limits, metrics and WebSocket caps are in process memory (per
  worker); fitted models are not cached. Multi-symbol analysis returns a blanket `503` when
  every symbol fails, instead of the root-cause status.
- **Library hygiene.** `mra_lib` still prints to stdout in places, re-wraps provider errors
  as `ValueError`, returns many results as dicts, and ships `types/protocols.py`, which nothing
  implements yet.
- **Data.** Alpha Vantage free tier: unadjusted daily prices, latest 100 bars, 25 requests/day.
  Yahoo Finance: 15m bars for the last 60 days only. No on-disk cache.

## Open Work

In progress:

- **Detector unification and library refactor** — one detector everywhere, stationary
  feature set, logging instead of `print`, typed results, and a decision on the protocols
  module.

Needs a decision:

- Which regime→strategy map is canonical (backtest vs. shared table).

Next:

- Deflated Sharpe / multiple-testing correction in the optimizer.
- Bar-by-bar regime attribution in the calibrator.
- Root-cause status for multi-symbol requests where every symbol fails.

Feature roadmap (none started; from the review's proposals):

- Model persistence and loading optimized/calibrated parameters (`--params file.json`).
- Regime history store and `/regimes/{symbol}/history`.
- Scheduled scanner with regime-change alerts.
- `mra backtest` command / backtest endpoint with a buy-and-hold benchmark.
- Multi-timeframe confirmation signal; explainability (per-state feature z-scores).
- Provider infrastructure: on-disk cache, `start`/`end` ranges; more providers (crypto, FRED).
- Regime-conditioned allocation and a paper-trading loop with a drawdown kill switch.

## Historical Notes

Earlier assessments (March 2026) reported backtest figures such as "+9.50% total return,
−57.09% excess return vs buy-and-hold, Sharpe 0.18". Those numbers were produced before the
October 2026 fixes to short accounting, metric formulas and walk-forward evaluation, and are
not reproducible with the current code. They are kept here only as history; rerun
`uv run mra-optimize` for current figures. Older planning documents are in
[archive/](archive/).
