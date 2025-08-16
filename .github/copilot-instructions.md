# Market Regime Analysis System — AI Coding Agent Instructions

## System Architecture

- Implements Jim Simons' Hidden Markov Model methodology for market regime detection and quantitative trading analysis (see `README.md`, `PLAN.md`).
- Core modules:
  - `market_regime_analysis/analyzer.py`: Main analysis engine, integrates HMM, technical analysis, arbitrage, risk management.
  - `market_regime_analysis/hmm_detector.py`: Hidden Markov Model implementation (Gaussian Mixture Models, transition matrices).
  - `market_regime_analysis/portfolio.py`: Multi-asset regime analysis, correlation, arbitrage pairs.
  - `market_regime_analysis/risk_calculator.py`: Kelly Criterion, regime multipliers, correlation adjustments.
  - `market_regime_analysis/data_classes.py`, `enums.py`: Data structures and enums for regimes, strategies, results.
- Interactive menu system in `main.py` (see `README.md` and `PLAN.md` for menu options and workflow).

## Developer Workflow

- Use Python 3.13+ (see `pyproject.toml`).
- Use `uv` for environment and package management:
  - Install dependencies: `uv sync`
  - Run scripts: `uv run main.py`, `uv run test_system.py`
- Tests: Place in root as `test_*.py`. Run with `uv run test_system.py` or `pytest`.
- Linting: Uses `ruff` (see `pyproject.toml`).
- Follow PEP 8 and project-specific style (see `pyproject.toml` for line length, quote style, import sorting).

## Key Patterns & Conventions

- All analysis flows through `MarketRegimeAnalyzer` and `PortfolioHMMAnalyzer` classes.
- Regime analysis returns a `RegimeAnalysis` dataclass (see `data_classes.py`).
- Risk management uses regime multipliers, confidence, persistence, and correlation adjustments (see `risk_calculator.py`, `PLAN.md`).
- Visualization: 5-panel chart system (see `plot_regime_analysis` in `analyzer.py`, `PLAN.md`).
- Statistical arbitrage: Z-score, autocorrelation, cross-asset pairs (see `analyzer.py`, `portfolio.py`).
- Exception handling and input validation are required for all user/data entry points.
- Type hints and docstrings are mandatory for all public methods/classes.

## Integration & External Dependencies

- Data providers: `yfinance`, `alpha-vantage`, `polygon` (see `providers/` package).
- Core dependencies: pandas, numpy, scikit-learn, matplotlib, yfinance, alpha-vantage.
- Optional: TA-Lib for technical indicators (fallbacks provided).

## Example Usage

```python
from market_regime_analysis import MarketRegimeAnalyzer, PortfolioHMMAnalyzer
analyzer = MarketRegimeAnalyzer("SPY")
analysis = analyzer.analyze_current_regime("1D")
analyzer.print_analysis_report("1D")
portfolio = PortfolioHMMAnalyzer(["SPY", "QQQ", "IWM"])
portfolio.print_portfolio_summary()
```

## References

- See `README.md` and `PLAN.md` for architecture, workflow, and implementation details.
- See `examples.py` for usage patterns and test cases.

---

**Update this file if project structure, workflow, or conventions change.**
