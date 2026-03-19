# mra-lib

Core library for market regime analysis using Hidden Markov Models. Contains all analysis logic with zero UI or framework dependencies.

## Key Modules

- `analyzer.py` — `MarketRegimeAnalyzer`, the main orchestrator
- `indicators/` — HMM-based regime detectors (GMM and hmmlearn)
- `backtesting/` — Strategy optimization, walk-forward validation, transaction costs
- `data_providers/` — Pluggable providers (Yahoo Finance, Alpha Vantage, Polygon.io)
- `risk/` — Kelly Criterion position sizing with regime adjustments
- `portfolio/` — Multi-asset correlation and regime analysis

## Usage

```python
from mra_lib import MarketRegimeAnalyzer, MarketRegime
from mra_lib.data_providers import MarketDataProvider
from mra_lib.backtesting import RegimeStrategy, BacktestEngine
```

Part of the [market-regime-analysis](../../README.md) workspace.
