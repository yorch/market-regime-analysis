# mra-cli

Click-based CLI for market regime analysis.

## Entry Points

- `mra` — Main CLI with subcommands (analysis, charts, export, monitoring, etc.)
- `mra-optimize` — Strategy optimization runner (grid/random search)

## Usage

```bash
uv run mra --help
uv run mra current-analysis --symbol SPY                    # yfinance (default), no key
uv run mra current-analysis --provider mock --symbol SPY    # offline synthetic data
uv run mra generate-charts --symbol SPY --output spy.png    # headless chart
uv run mra continuous-monitoring --symbol SPY --once        # single refresh
uv run mra-optimize --mode grid --symbol SPY --provider yfinance
```

- `--provider` / `--api-key` are accepted on the group (`mra --provider X <cmd>`) or the
  subcommand (`mra <cmd> --provider X`); provider choices come from the provider registry.
- The default provider is `yfinance`; override with the `DEFAULT_PROVIDER` environment variable.
- API keys are resolved lazily from the flag or environment (`ALPHA_VANTAGE_API_KEY`,
  `POLYGON_API_KEY`, `APCA_API_KEY_ID` + `APCA_API_SECRET_KEY`, `TIINGO_API_KEY`), and only by
  commands that fetch data.
- Commands exit with status 1 on failure; `--debug` shows the full traceback.

See `uv run mra --help` for all available commands.

Part of the [market-regime-analysis](../../README.md) workspace.
