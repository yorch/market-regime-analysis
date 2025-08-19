# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the `market-regime-analysis` project.

## Project Overview

The `market-regime-analysis` project is a professional-grade Python application that implements Jim Simons' Hidden Markov Model (HMM) methodology for market regime detection and quantitative trading analysis. The system provides multi-timeframe analysis (daily, hourly, and 15-minute), statistical arbitrage opportunity identification, and risk management using the Kelly Criterion.

The project is structured as a command-line application using the `click` library, with the core logic encapsulated in the `market_regime_analysis` package. The main components are:

* **`MarketRegimeAnalyzer`**: The main analysis engine that integrates HMM detection with technical analysis, statistical arbitrage, and risk management.
* **`HiddenMarkovRegimeDetector`**: The core HMM implementation using Gaussian Mixture Models to identify hidden market states.
* **`PortfolioHMMAnalyzer`**: A class for analyzing a portfolio of multiple assets.
* **`SimonsRiskCalculator`**: A utility for calculating position sizing based on the Kelly Criterion and regime-adjusted multipliers.
* **Data Providers**: A set of classes for fetching market data from different sources, such as Yahoo Finance, Alpha Vantage, and Polygon.

## Building and Running

The project uses `uv` for dependency management. The key commands are:

* **Install dependencies:** `uv sync`
* **Run the CLI:** `uv run main.py --help`
* **Run tests:** `uv run test_system.py` or `uv run pytest`

### CLI Usage Examples

* **Run a quick analysis using Yahoo Finance:**

    ```bash
    uv run main.py current-analysis --provider yfinance --symbol SPY
    ```

* **Run a detailed analysis with a specific timeframe and provider:**

    ```bash
    uv run main.py detailed-analysis --symbol SPY --timeframe 1D --provider alphavantage --api-key YOUR_KEY
    ```

* **Generate charts:**

    ```bash
    uv run main.py generate-charts --symbol SPY --timeframe 1D --days 60 --provider yfinance
    ```

## Development Conventions

* **Linting and Formatting:** The project uses `ruff` for linting and formatting. The configuration is in the `pyproject.toml` file.
* **Testing:** The project uses `pytest` for testing. Tests are located in the `test_system.py` and `test_mock.py` files.
* **Type Hinting:** The code uses type hints throughout.
* **Docstrings:** The code is well-documented with docstrings.
* **Modularity:** The code is organized into modules with clear responsibilities.
* **Data Classes and Enums:** The project uses data classes and enums to represent data structures and predefined values.
