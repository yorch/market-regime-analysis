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

## Strategy optimization (`mra-optimize`)

```bash
uv run mra-optimize --mode random --iterations 50 --seed 42 --holdout-frac 0.2 \
    --symbol SPY --provider yfinance --output results.json
```

- `--holdout-frac` (default 0.2) withholds the most recent bars from the search.
  The best parameters are reported twice: **IN-SAMPLE** (on the search period they
  were selected on — optimistic, with the number of trials) and **HOLDOUT
  (OUT-OF-SAMPLE)**. Use `--holdout-frac 0` to disable.
- `--seed` makes random search reproducible; `--output` is written in both grid
  and random modes.
- Provider API keys are read from the environment (`ALPHA_VANTAGE_API_KEY`,
  `POLYGON_API_KEY`, `TIINGO_API_KEY`, `APCA_API_KEY_ID` + `APCA_API_SECRET_KEY`).

Part of the [market-regime-analysis](../../README.md) workspace.
