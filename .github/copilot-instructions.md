# Market Regime Analysis System — AI Coding Agent Instructions

## System Architecture

- Implements Jim Simons' Hidden Markov Model methodology for market regime detection and quantitative trading analysis (see `README.md`, `PLAN.md`).
- Multi-package workspace: `mra_lib` (core library), `mra_cli` (CLI), `mra_web` (FastAPI API).
- Core modules:
  - `packages/mra_lib/src/mra_lib/analyzer.py`: Main analysis engine, integrates HMM, technical analysis, arbitrage, risk management.
  - `packages/mra_lib/src/mra_lib/indicators/hmm_detector.py`: Hidden Markov Model implementation (Gaussian Mixture Models, transition matrices).
  - `packages/mra_lib/src/mra_lib/portfolio/portfolio.py`: Multi-asset regime analysis, correlation, arbitrage pairs.
  - `packages/mra_lib/src/mra_lib/risk/risk_calculator.py`: Kelly Criterion, regime multipliers, correlation adjustments.
  - `packages/mra_lib/src/mra_lib/config/data_classes.py`, `config/enums.py`: Data structures and enums for regimes, strategies, results.
- CLI entry point: `packages/mra_cli/src/mra_cli/main.py` (command: `uv run mra`).
- API entry point: `packages/mra_web/src/mra_web/` (command: `uv run mra-api`).

## Developer Workflow

- Use Python 3.13+ (see `pyproject.toml`).
- Use `uv` for environment and package management:
  - Install dependencies: `uv sync`
  - Run CLI: `uv run mra`; Run API: `uv run mra-api --dev`
- Tests: Place in `packages/mra_lib/tests/`, `packages/mra_cli/tests/`, `packages/mra_web/tests/`. Run with `uv run pytest`.
- Linting: `uv run ruff check packages/ examples/`; Formatting: `uv run ruff format packages/ examples/`; Type checking: `uv run mypy packages/`.
- Task runner: Use `just` commands where available.
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

- Data providers: `yfinance`, `alpha-vantage`, `polygon` (see `packages/mra_lib/src/mra_lib/data_providers/` package).
- Core dependencies: pandas, numpy, scikit-learn, matplotlib, yfinance, alpha-vantage.
- Optional: TA-Lib for technical indicators (fallbacks provided).

## Example Usage

```python
from mra_lib import MarketRegimeAnalyzer, PortfolioHMMAnalyzer
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
