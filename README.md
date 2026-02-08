# qlab

A Python library for researching and backtesting US equity trading strategies.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import qlab
from qlab.data.csv_provider import generate_synthetic
from qlab.features import simple_returns
from qlab.alphas import momentum
from qlab.portfolio import equal_weight_long_short, normalize_weights
from qlab.backtest import run_backtest, BacktestConfig
from qlab.risk import performance_summary

# 1. Generate synthetic price data
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META",
           "NVDA", "TSLA", "JPM", "V", "JNJ"]
prices = generate_synthetic(tickers, start="2018-01-02", end="2023-12-29", seed=42)
close = prices["adj_close"]

# 2. Compute a momentum signal
signal = momentum(close, lookback=252, skip=21)

# 3. Construct a long-short portfolio
weights = equal_weight_long_short(signal.dropna(), long_pct=0.2, short_pct=0.2)
weights = normalize_weights(weights, gross_exposure=2.0, net_exposure=0.0)

# 4. Backtest
config = BacktestConfig(
    rebalance_freq="monthly",
    commission_bps=5,
    slippage_bps=5,
    signal_lag=1,
    execution_price="open",
)
result = run_backtest(weights, prices, config=config)

# 5. Analyse performance
summary = performance_summary(result.portfolio_returns)
for k, v in summary.items():
    print(f"{k:25s}: {v:>10.4f}")
```

## Package structure

```
qlab/
├── data/           Market data ingestion and caching
├── features/       Returns, volatility, rolling stats, cross-sectional transforms
├── alphas/         Reusable alpha signal functions
├── portfolio/      Signal-to-weight mapping and constraints
├── backtest/       Daily-bar backtesting engine
├── risk/           Performance metrics, drawdowns, factor regression
└── utils/          Calendars, alignment, validation
```

## Running tests

```bash
pytest
```

## Data model

All data uses a stacked `pandas.DataFrame` or `Series` with a two-level
`MultiIndex(date, ticker)`.  See [DESIGN.md](DESIGN.md) for rationale.
