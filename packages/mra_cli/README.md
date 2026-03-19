# mra-cli

Click-based CLI for market regime analysis.

## Entry Points

- `mra` — Main CLI with subcommands (analysis, charts, export, monitoring, etc.)
- `mra-optimize` — Strategy optimization runner (grid/random search)

## Usage

```bash
uv run mra --help
uv run mra current-analysis --symbol SPY --provider yfinance
uv run mra-optimize --mode grid --symbol SPY --provider yfinance
```

See `uv run mra --help` for all available commands.

Part of the [market-regime-analysis](../../README.md) workspace.
