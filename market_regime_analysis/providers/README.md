# Market Data Providers

This package contains the plug-and-play architecture for market data providers in the Market Regime Analysis system.

## Structure

```
providers/
├── __init__.py              # Auto-discovery and registration
├── base.py                  # Base interfaces and configuration
├── yfinance_provider.py     # Yahoo Finance provider
├── alphavantage_provider.py # Alpha Vantage provider
├── mock_provider.py         # Mock data for testing
├── example_new_provider.py  # Example of adding new providers
└── README.md               # This file
```

## Available Providers

- **YFinance**: Free Yahoo Finance data with comprehensive market coverage
- **Alpha Vantage**: Professional API with real-time and historical data (requires API key)
- **Mock**: Synthetic data for testing and development
- **Example**: Template showing how to add new providers

## Adding New Providers

Adding a new provider is extremely simple:

### 1. Create Provider File

Create a new file in this directory (e.g., `my_provider.py`):

```python
from typing import ClassVar
import pandas as pd
from .base import MarketDataProvider

class MyProvider(MarketDataProvider):
    provider_name = "myprovider"
    supported_intervals: ClassVar[set[str]] = {"1d", "1h", "15m"}
    supported_periods: ClassVar[set[str]] = {"1mo", "1y"}
    requires_api_key = True  # or False
    rate_limit_per_minute = 60
    description = "My custom data provider"

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        # Your implementation here
        # Must return DataFrame with Open, High, Low, Close, Volume columns
        pass
```

### 2. Register Provider

Add to `__init__.py`:

```python
from .my_provider import MyProvider
MarketDataProvider.register(MyProvider)
```

### 3. Use Immediately

Your provider is now available throughout the system:

```bash
# CLI usage
uv run main.py list-providers  # Shows your provider
uv run main.py detailed-analysis --provider myprovider --symbol SPY

# Python usage
from market_regime_analysis.providers import create_provider
provider = create_provider("myprovider", api_key="your_key")
```

## Configuration

Providers can accept configuration via `ProviderConfig`:

```python
from market_regime_analysis.providers import create_provider

provider = create_provider("alphavantage",
                         api_key="your_key",
                         timeout=60,
                         retries=5,
                         custom_param="value")
```

## Best Practices

1. **Lazy Imports**: Import external libraries inside methods to avoid import errors if the library isn't installed
2. **Error Handling**: Provide clear error messages for API failures, missing keys, etc.
3. **Validation**: Use `self.validate_parameters()` to check supported intervals/periods
4. **Standardization**: Always call `self.standardize_dataframe()` before returning data
5. **Documentation**: Include comprehensive docstrings and examples

## Provider Metadata

Each provider should define these class attributes:

- `provider_name`: Unique identifier (string)
- `supported_intervals`: Set of supported time intervals
- `supported_periods`: Set of supported time periods
- `requires_api_key`: Whether API key is required (boolean)
- `rate_limit_per_minute`: API rate limit (integer)
- `description`: Human-readable description (string)
