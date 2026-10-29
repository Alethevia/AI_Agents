# AI Agents: experimental backtesting framework

The default branch now includes the implementation previously left on the development branch. This is research code, with synthetic examples; it is not a validated production trading service.

## Run the offline example

Python 3.11+:

```bash
python -m pip install -r requirements-core.txt
python -m pytest tests -q -o addopts=
python -m examples.backtest_example
```

The example generates artificial OHLCV prices using a fixed seed and date, runs a moving-average strategy, and writes charts and a report under `backtest_output/`. These outputs test mechanics, not investment performance. The full optional infrastructure dependencies are in `pyproject.toml`; broker, database, API, Transformer and RL integrations have not all been exercised.

The 2026-10-29 repair corrects chronological event ordering, makes orders wait for a later quote, uses existing shares for exits, avoids charging fill slippage twice, and repairs the example's reporting calls. See [AUDIT.md](AUDIT.md) for validation and remaining limits. There is no verified real-market return, originality, or institutional-quality claim.
