# mra-lib

Core library for market regime analysis using Hidden Markov Models. Contains all analysis logic and no CLI or web framework dependencies (it still prints reports and uses matplotlib for charts).

## Key Modules

- `analyzer.py` — `MarketRegimeAnalyzer`, the main orchestrator
- `indicators/` — the regime model: `TrueHMMDetector`, shared causal features, state → regime mapping
- `backtesting/` — Strategy optimization, walk-forward validation, transaction costs
- `data_providers/` — Pluggable providers (Yahoo Finance, Alpha Vantage, Polygon.io, Alpaca, Tiingo, offline mock)
- `risk/` — Kelly Criterion position sizing with regime adjustments
- `portfolio/` — Multi-asset correlation and regime analysis

## Usage

```python
from mra_lib import MarketRegimeAnalyzer, MarketRegime
from mra_lib.data_providers import MarketDataProvider
from mra_lib.backtesting import RegimeStrategy, BacktestEngine
```

## Regime model

`TrueHMMDetector` is the only model; the analyzer, `regime-forecast`, walk-forward
validation and the optimizer all use it (`HiddenMarkovRegimeDetector` is a deprecated
alias that emits `DeprecationWarning`).

- **Features** (`indicators/features.build_hmm_features`): `log_return`,
  `log_volatility` (20 bars), `vol_expansion` (`log(vol_10 / vol_40)`), `trend_strength`
  (`(SMA_10 − SMA_30) / close`), `autocorr_1` (30 bars) and `log_volume_ratio`. All are
  causal (truncating future bars never changes past rows) and scale-free (no price
  levels); the first 40 bars are warm-up. Bars without volume get a neutral ratio,
  decided per bar.
- **Model**: Gaussian HMM, `covariance_type="diag"` (configurable), `min_covar=1e-3`,
  `n_init=10` EM restarts keeping the best log-likelihood, `n_iter=200`. `fit` refuses
  data with fewer than 1.5 feature rows per free parameter (a 6-state model needs 161
  rows ≈ 201 bars). With shorter history (e.g. Alpha Vantage's 100 free daily bars) it
  fits the largest state count the data supports and logs a warning
  (`adapt_n_states=False` raises instead). It also logs a warning if EM did not converge
  or its log-likelihood decreased. Walk-forward validation uses the same model with
  `hmm_n_init=3` restarts per refit to bound runtime. `select_n_states(df, candidates=range(2, 7))` picks a state count by BIC;
  the default stays at 6 states.
- **State order and labels**: states are sorted by volatility after fitting, so state
  ints are stable whenever refits reach the same optimum. Each state's mean is
  de-standardized and labelled by a fixed decision tree (`indicators/regime_mapping.py`):
  relative volatility ≥ 1.4× the sample's typical level → High Volatility; expanding
  volatility in an up-trend → Breakout; `|trend_strength / vol|` ≥ 1.0 → Bull/Bear
  Trending; relative volatility ≤ 1/1.4 → Low Volatility; lag-1 autocorrelation ≤ −0.1
  → Mean Reverting; otherwise Unknown.
- **Posteriors**: per-bar history and confidence use filtered posteriors
  `P(s_t | o_1..o_t)` (`filtered_posteriors`, `regime_history`, `predict_with_states`);
  `predict_regime` returns the filtered argmax at the last bar (`use_viterbi=True` for
  the Viterbi path's last state).

## Backtesting conventions

- **Accounting**: cash-based. Longs pay notional + costs on entry; shorts receive
  sale proceeds (less costs) on entry and pay to cover on exit. Equity = cash +
  long market value − short liability, so a flat-price round trip at zero cost
  ends exactly at the starting capital for both directions.
- **Equity curve**: one point per bar, marked at the close; the last point
  includes the forced close of any open position (with exit costs).
- **Fills**: signals execute at the bar's close. Stops/take-profits trigger on
  the next bars' high/low and are gap-aware (a long stop fills at
  `min(open, stop)`, a short stop at `max(open, stop)`).
- **Costs** (`TransactionCostModel`): `spread_bps` is the full quoted spread and
  half is charged per side; slippage per side; market impact
  (`coeff * sqrt(shares / avg_volume)`) uses a 20-bar rolling average volume.
- **Metrics** (`PerformanceMetrics`, `periods_per_year=252` by default):
  Sharpe = mean(r − rf/ppy) / std(r) · √ppy; Sortino uses downside deviation
  √mean(min(r − rf/ppy, 0)²) over all observations; Calmar = CAGR / |max DD|
  (negative for losing strategies). Ratios with a zero denominator are 0.0.
  Trade stats come from `backtesting/trade_stats.py`: profit factor is `inf`
  with no losing trades and `nan` when undefined (no trades).
- **Walk-forward**: Sharpe and max drawdown are computed on the stitched
  per-bar equity curve across windows (partial final windows count by their
  length). A failed HMM refit keeps the previous model and logs a warning.
  `WalkForwardValidator.compute_regimes()` caches regimes per window so many
  strategy parameter sets can be evaluated without refitting the HMM.
- **Optimization honesty**: `StrategyOptimizer(..., holdout_frac=0.2)` withholds
  the most recent bars from the search; `evaluate_holdout(params)` reports the
  chosen parameters on that untouched segment. Search results are in-sample with
  respect to parameter selection (`n_trials` is reported). The calibrator
  supports the same `holdout_frac`; otherwise `CalibrationResult.in_sample` is True.

Part of the [market-regime-analysis](../../README.md) workspace.
