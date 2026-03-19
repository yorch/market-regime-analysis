# Project Assessment — Market Regime Analysis

**Date**: 2026-03-19
**Assessed by**: Claude Code

---

## What This Project Does

A market regime analysis system that uses Hidden Markov Models to classify market states (Bull/Bear Trending, Mean Reverting, High/Low Volatility, Breakout) and generate trading signals. Built with professional infrastructure:

- **CLI** (Click, 8 commands) + **REST API** (FastAPI with JWT auth + WebSocket)
- **3 data providers** (Alpha Vantage, Polygon.io, Yahoo Finance) via plug-and-play architecture
- **Multi-timeframe analysis** (daily, hourly, 15-min)
- **Backtesting engine** with walk-forward validation and transaction cost modeling
- **Strategy optimizer** with grid search, random search, and composite scoring
- **Kelly Criterion position sizing** from backtest results
- **Portfolio analysis** across multiple symbols

---

## What's Done Well

| Area | Rating | Notes |
|------|--------|-------|
| **Code quality** | 8/10 | Clean, typed, well-structured Python 3.13+ |
| **Architecture** | 8/10 | Excellent provider plugin system, good separation of concerns |
| **Documentation** | 9/10 | 14 markdown files, 3,500+ lines, including a professional trading review |
| **Infrastructure** | 9/10 | CLI, API, CI/CD, Docker, backtester, optimizer all complete |
| **Bug awareness** | High | 7 critical bugs documented and fixed (capital tracking, look-ahead bias, GMM→HMM) |
| **Optimization framework** | 7/10 | Grid/random search with walk-forward validation and composite scoring (new) |
| **CI pipeline** | 8/10 | Ruff lint + format checks, pytest, caching, API key secrets |

---

## What's Broken / Incomplete

1. **Strategy still underperforms buy-and-hold** — Best optimized result: +9.50% total return vs -57.09% excess return. Sharpe improved from -0.15 to 0.18 but remains weak.
2. ~~**Arbitrary parameters**~~ **Partially addressed** — Optimizer framework exists (grid/random search), but results show the strategy cannot beat buy-and-hold even with tuned parameters. The problem may be structural, not just parametric.
3. **Statistical arbitrage is naive** — Simple Z-score > 2 threshold, no cointegration testing. Portfolio module promises "statistical arbitrage pair identification" in docstring but doesn't implement it.
4. ~~**Portfolio analysis uses price correlation**~~ **Fixed** — Now correctly uses returns-based correlation via `pct_change()`.
5. **Test coverage improved but gaps remain** — ~1,000 lines across 6 test files (up from ~500), but only `test_engine.py` (22 assertions) and `test_strategy.py` (27 assertions) use proper pytest assertions. `test_backtest.py`, `test_mock.py`, `test_system.py`, and `test_true_hmm.py` still return booleans or print output without assertions.
6. **Old GMM detector still ships** alongside the proper HMM — `hmm_detector.py` (GaussianMixture) used by main analyzer, `true_hmm_detector.py` (hmmlearn) used by backtester. Confusing dual implementation.
7. **No model persistence** — Retrains from scratch every time. No save/load/pickle functionality.
8. **No drawdown circuit breaker** — Per-trade stop-loss exists, but no portfolio-level max drawdown killswitch or position correlation limits.

---

## What Changed Since Last Assessment (2026-03-18)

- **Phase 1 partially complete**: Strategy optimization framework built (grid search, random search, walk-forward validation, composite scoring)
- **New test files**: `test_engine.py` (326 lines, 22 assertions) and `test_strategy.py` (147 lines, 27 assertions) added with proper pytest assertions
- **Backtester hardened**: Direction propagation fixes, base_position_fraction wired through, robustness improvements
- **CI formatting fixed**: `test_engine.py` reformatted to pass `ruff format --check`
- **Portfolio correlation fixed**: Now uses returns-based correlation (was price-based)
- **Documentation updated**: CLAUDE.md and README.md updated with backtester documentation

---

## What's Next (Priority Order)

### Phase 1 — Make the Strategy Work (IN PROGRESS)

- ~~Grid search / optimization of regime thresholds and multipliers~~ Done — framework built
- ~~Walk-forward validation~~ Done — anchored/rolling walk-forward with HMM retraining
- **Investigate structural issues**: Current results (-57% excess return) suggest the regime-based approach may need fundamental changes, not just parameter tuning
- Consider alternative signal generation (momentum, trend-following filters, regime transition signals)
- Test across multiple market cycles and symbols
- Target Sharpe > 0.5 before anything else

### Phase 2 — Harden the System

- Convert `test_backtest.py`, `test_mock.py`, `test_system.py`, `test_true_hmm.py` to proper pytest assertions
- Remove or deprecate the old GMM detector (`hmm_detector.py`)
- Add model serialization (save/load trained HMM via joblib/pickle)
- Implement portfolio-level drawdown killswitch
- Add test coverage reporting to CI

### Phase 3 — Improve Statistical Rigor

- Cointegration testing (Engle-Granger / Johansen) for pairs trading
- ~~Returns-based correlation in portfolio analysis~~ Done
- Regime threshold optimization tied to backtest performance
- Out-of-sample validation across different market environments

### Phase 4 — Production Readiness

- Paper trading for 3-6 months
- Persistent trade logging / audit trail
- Alerting system for regime changes
- Performance monitoring dashboard

---

## Bottom Line

The **infrastructure is professional-grade** and now includes a complete **optimization and walk-forward validation framework**. However, the **trading strategy still cannot beat buy-and-hold** — even after parameter optimization, excess return is deeply negative (-57%). This suggests the issue may be structural rather than parametric: the regime detection → position sizing pipeline may need fundamental rethinking, not just better parameters. The immediate priority shifts from "tune parameters" to "investigate why regime-based signals don't generate alpha" before investing further in infrastructure.
