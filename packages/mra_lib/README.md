# mra-lib

Core library for market regime analysis using Hidden Markov Models. Contains all analysis logic and no CLI or web framework dependencies (it still prints reports and uses matplotlib for charts).

## Key Modules

- `analyzer.py` — `MarketRegimeAnalyzer`, the main orchestrator
- `indicators/` — HMM-based regime detectors (GMM and hmmlearn)
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
