# Comprehensive Prompt to Recreate Jim Simons' Market Regime Analysis System

## Overview

Create a professional-grade Python application that implements Jim Simons' Hidden Markov Model methodology for market regime detection and quantitative trading analysis, following the exact approach used by Renaissance Technologies.

## Core Requirements

### 1. System Architecture

```
Market Regime Analysis System
├── Hidden Markov Model Implementation
├── Multi-Timeframe Analysis Engine
├── Statistical Arbitrage Detection
├── Risk Management Calculator
├── Portfolio Analysis Tools
└── Interactive User Interface
```

### 2. Dependencies and Imports

```python
# Required libraries
pandas, numpy, yfinance, scikit-learn, matplotlib
datetime, typing, time, os, dataclasses, enum

# Optional: TA-Lib (with fallback implementations)
# Scientific: GaussianMixture, StandardScaler
# Visualization: pyplot, ListedColormap
```

## Class Structure Implementation

### 3. Enumerations and Data Classes

#### MarketRegime Enum

```python
class MarketRegime(Enum):
    BULL_TRENDING = "Bull Trending"
    BEAR_TRENDING = "Bear Trending"
    MEAN_REVERTING = "Mean Reverting"
    HIGH_VOLATILITY = "High Volatility"
    LOW_VOLATILITY = "Low Volatility"
    BREAKOUT = "Breakout"
    UNKNOWN = "Unknown"
```

#### TradingStrategy Enum

```python
class TradingStrategy(Enum):
    TREND_FOLLOWING = "Trend Following"
    MOMENTUM = "Momentum"
    MEAN_REVERSION = "Mean Reversion"
    STATISTICAL_ARBITRAGE = "Statistical Arbitrage"
    VOLATILITY_TRADING = "Volatility Trading"
    DEFENSIVE = "Defensive"
    AVOID = "Avoid Trading"
```

#### RegimeAnalysis DataClass

```python
@dataclass
class RegimeAnalysis:
    current_regime: MarketRegime
    hmm_state: int
    transition_probability: float
    regime_persistence: float
    recommended_strategy: TradingStrategy
    position_sizing_multiplier: float
    risk_level: str
    arbitrage_opportunities: List[str]
    statistical_signals: List[str]
    key_levels: Dict[str, float]
    regime_confidence: float
```

## Core Implementation Classes

### 4. HiddenMarkovRegimeDetector Class

**Purpose**: True HMM implementation following Simons' mathematical approach

**Key Methods**:

- `__init__(n_states=6)`: Initialize with 6 market states
- `_prepare_features(df)`: Extract mathematical features (returns, volatility, skewness, kurtosis, autocorrelation, cross-correlations)
- `fit(df)`: Train HMM using Gaussian Mixture Models with transition matrix estimation
- `_estimate_transition_matrix(states)`: Calculate state transition probabilities
- `_map_states_to_regimes(X, states)`: Map mathematical states to interpretable market regimes
- `predict_regime(df)`: Return (regime, state, confidence)
- `get_transition_probability(current, target)`: State transition likelihood
- `calculate_regime_persistence(state, lookback=20)`: Regime stability metric

**Mathematical Features**:

- Returns and log returns
- Rolling skewness and kurtosis (higher order moments)
- Volatility and ATR normalization
- Trend strength (moving average differences)
- Volume ratios (when available)
- Cross-correlations between features
- All features standardized using StandardScaler

### 5. MarketRegimeAnalyzer Class (Main Class)

**Purpose**: Primary analysis engine implementing full Simons methodology

**Initialization**:

```python
def __init__(symbol="SPY", periods={
    "1D": "2y",    # Daily data for 2 years
    "1H": "6mo",   # Hourly data for 6 months
    "15m": "2mo"   # 15-min data for 2 months
})
```

**Core Methods**:

#### Data Management

- `_load_data()`: Fetch data using yfinance for all timeframes
- `_calculate_technical_indicators(df)`: Comprehensive technical analysis
- `_calculate_indicators()`: Process all timeframes
- `_train_hmm_models()`: Train HMM for each timeframe

#### Technical Indicators (with TA-Lib fallbacks)

- EMAs (9, 34 periods)
- ATR and normalized ATR
- Bollinger Bands
- RSI, MACD
- Volume analysis
- **Statistical Arbitrage Features**:
  - Price Z-scores
  - Return autocorrelation
  - Mean reversion signals

#### Regime Analysis

- `analyze_current_regime(timeframe)`: Main analysis function returning RegimeAnalysis
- `_get_trading_strategy(regime)`: Map regimes to strategies
- `_get_position_sizing_multiplier(regime, confidence)`: Risk-adjusted sizing
- `_identify_arbitrage_opportunities(df)`: Core Simons statistical arbitrage
- `_generate_statistical_signals(df, regime)`: Regime-specific signals
- `_identify_key_levels(df)`: Support/resistance with multiple methods

#### Reporting and Visualization

- `print_analysis_report(timeframe)`: Comprehensive formatted report
- `plot_regime_analysis(timeframe, days)`: 5-panel chart with regime background
- `run_continuous_monitoring(interval)`: Real-time monitoring with memory management
- `export_analysis_to_csv(filename)`: Data export for backtesting

### 6. Portfolio Analysis Classes

#### PortfolioHMMAnalyzer Class

**Purpose**: Multi-asset regime analysis following Renaissance approach

**Key Methods**:

- `calculate_regime_correlations()`: Cross-asset regime correlations
- `get_portfolio_regime_summary()`: Portfolio-level metrics
- `identify_arbitrage_pairs()`: Statistical arbitrage pairs detection
- `print_portfolio_summary()`: Comprehensive portfolio analysis

**Statistical Arbitrage Implementation**:

- Price correlation analysis
- Cointegration proxy (spread stationarity)
- Z-score divergence detection
- Opportunity ranking by strength

#### SimonsRiskCalculator Class

**Purpose**: Professional risk management following Renaissance approach

**Static Methods**:

- `calculate_kelly_optimal_size(win_rate, avg_win, avg_loss, confidence)`: Kelly Criterion with confidence scaling
- `calculate_regime_adjusted_size(base_size, regime, confidence, persistence)`: Multi-factor position sizing
- `calculate_correlation_adjusted_size(base_size, correlation)`: Correlation-based adjustments

## Critical Implementation Details

### 7. Jim Simons Methodology Compliance

#### True Hidden Markov Models

- Use Gaussian Mixture Models as HMM approximation
- Implement proper transition matrices
- Calculate state persistence metrics
- Multi-feature mathematical analysis (not just price/volume)

#### Statistical Arbitrage (Core Simons Strategy)

- Z-score mean reversion analysis
- Autocorrelation breakdown detection
- Cross-asset pairs trading identification
- Statistical significance thresholds

#### Mathematical Sophistication

- Higher-order moments (skewness, kurtosis)
- Cross-correlations between features
- Regime transition probabilities
- Confidence-weighted decision making

### 8. Risk Management (Renaissance Approach)

#### Position Sizing Multipliers

```python
regime_multipliers = {
    BULL_TRENDING: 1.3,
    BEAR_TRENDING: 0.7,
    MEAN_REVERTING: 1.2,
    HIGH_VOLATILITY: 0.4,
    LOW_VOLATILITY: 1.1,
    BREAKOUT: 0.9,
    UNKNOWN: 0.2
}
```

#### Multi-Factor Adjustments

- Base regime multiplier
- Confidence scaling (0.3 to 1.0)
- Persistence adjustment (0.7 to 1.0)
- Correlation reduction for similar positions
- Safety caps (1% to 50% maximum)

### 9. User Interface Requirements

#### Interactive Menu System

1. Current HMM Regime Analysis (All Timeframes)
2. Detailed HMM Analysis (Single Timeframe)
3. Generate HMM Charts
4. Export HMM Analysis to CSV
5. Start Continuous HMM Monitoring
6. Multi-Symbol HMM Analysis
7. Exit

#### Comprehensive Reporting

- HMM state and confidence
- Regime persistence and transition probabilities
- Statistical arbitrage opportunities
- Strategy-specific recommendations
- Risk-adjusted position sizing
- Key support/resistance levels

### 10. Visualization Requirements

#### 5-Panel Chart System

1. **Price with Regime Background**: Color-coded regime periods
2. **Statistical Arbitrage Signals**: Z-scores with overbought/oversold levels
3. **Volatility Measures**: ATR and rolling volatility
4. **Return Autocorrelation**: Momentum indicator
5. **HMM State Sequence**: Time series of regime classifications

#### Chart Features

- Regime color-coding background
- Technical indicator overlays
- Statistical signal markers
- Professional formatting with legends and grids

## Advanced Features Implementation

### 11. Memory Management and Performance

- Clear data structures in continuous monitoring
- Efficient feature calculation
- Error handling for all edge cases
- Minimum data requirements (50+ bars)

### 12. Data Export and Integration

- CSV export with all HMM metrics
- Timestamp and confidence tracking
- Arbitrage opportunity logging
- Backtesting-ready format

### 13. Error Handling and Validation

- Internet connectivity checks
- Data availability validation
- Insufficient data warnings
- Graceful failure handling
- User input validation

## Example Usage Patterns

### 14. Basic Implementation

```python
# Single symbol analysis
analyzer = MarketRegimeAnalyzer("SPY")
analysis = analyzer.analyze_current_regime("1D")
analyzer.print_analysis_report("1D")

# Portfolio analysis
portfolio = PortfolioHMMAnalyzer(["SPY", "QQQ", "IWM"])
portfolio.print_portfolio_summary()

# Risk calculation
size = SimonsRiskCalculator.calculate_regime_adjusted_size(
    base_size=0.02, regime=regime, confidence=0.85, persistence=0.70
)
```

### 15. Professional Features

- Continuous monitoring with auto-refresh
- Multi-timeframe correlation analysis
- Statistical arbitrage pair identification
- Kelly-optimal position sizing
- Confidence-weighted strategies

## Quality Standards

### 16. Code Quality Requirements

- Type hints throughout
- Comprehensive docstrings
- Error handling for all edge cases
- Memory-efficient data structures
- Professional variable naming
- Modular, extensible design
- Each class and method should have a single responsibility
- Unit tests for all critical components
- Organized code into logical modules

### 17. Mathematical Accuracy

- Proper statistical implementations
- Numerical stability considerations
- Edge case handling (division by zero, etc.)
- Scientific computing best practices
- Validation against known benchmarks
