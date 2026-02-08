# qlab

A Python-based quantitative research platform for US equities. Designed for systematic alpha research, portfolio construction, backtesting, and stock selection using real market data.

## Overview

**qlab** provides a modular, transparent framework for:

- Computing and combining cross-sectional alpha signals
- Constructing constrained portfolios with sector caps and position limits
- Backtesting strategies with realistic transaction cost modeling
- Running rule-based stock selection pipelines on live market data
- Interactive visualization through a web-based GUI

**What it is:** A research and prototyping tool for quantitative analysts, students, and developers exploring systematic equity strategies.

**What it is not:** A production trading system, a black-box signal generator, or investment advice. All outputs are for research and educational purposes only.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        GUI Frontend                      │
│               (React / TypeScript / Tailwind)            │
├─────────────────────────────────────────────────────────┤
│                      FastAPI Backend                     │
│         /api/backtest  /api/alphas  /api/selection       │
├─────────────────────────────────────────────────────────┤
│                  StockPicker Pipeline                    │
│        (app/stock_picker.py — CLI + API modes)          │
├─────────────────────────────────────────────────────────┤
│                      Core Library                        │
│   data → features → alphas → portfolio → backtest → risk│
└─────────────────────────────────────────────────────────┘
```

**Design philosophy:**

- **Stacked MultiIndex** — All data uses `MultiIndex(date, ticker)`. Cross-sectional operations use `groupby(level="date")`, time-series use `groupby(level="ticker")`.
- **Pure functions** — Alpha signals, portfolio weights, and risk metrics are stateless functions operating on pandas Series/DataFrames.
- **Explicit configuration** — All parameters are passed through frozen dataclasses. No hidden global state.
- **Reproducibility** — Deterministic seeds, cached data, and versioned configs per run.

## Repository Structure

```
qlab/
├── qlab/                    Core research library
│   ├── data/                Market data providers (yfinance, CSV, Parquet cache)
│   ├── features/            Returns, volatility, rolling stats, cross-sectional transforms
│   ├── alphas/              Reusable alpha signal functions (momentum, mean reversion, etc.)
│   ├── portfolio/           Signal-to-weight mapping, constraints (position limits, turnover)
│   ├── backtest/            Daily-bar backtesting engine with transaction cost modeling
│   ├── risk/                Performance metrics, drawdowns, factor regression
│   └── utils/               Trading calendars, index alignment, validation
│
├── app/                     Backend research engine
│   ├── config.py            SelectionConfig dataclass (all pipeline parameters)
│   ├── stock_picker.py      Multi-factor stock selection pipeline
│   └── run_selection.py     CLI entrypoint for running selections
│
├── gui/                     Web-based GUI
│   ├── backend/             FastAPI application (routes, services, models)
│   └── frontend/            React + TypeScript + Tailwind (dark theme)
│
├── docs/tutorials/          Jupyter notebooks (6 tutorials with real data)
├── tests/                   Unit tests (pytest)
├── universe.csv             S&P 500 universe with GICS sector assignments
└── pyproject.toml           Package configuration
```

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/licongxu/qlab.git
cd qlab

# Create environment (venv or conda)
python -m venv .venv && source .venv/bin/activate
# or: conda create -n qlab python=3.12 && conda activate qlab

# Install with dev dependencies
pip install -e ".[dev]"

# Verify installation
pytest
```

**Optional — GPU acceleration:**
Some notebooks demonstrate ML workflows. To use GPU:

```bash
export CUDA_VISIBLE_DEVICES=1  # Select GPU device
```

**Frontend dependencies (for GUI only):**

```bash
cd gui/frontend && npm install && cd ../..
```

## Using the Platform

### A) As a Python Library

Import modules directly for custom research scripts:

```python
from qlab.data import YFinanceProvider, ParquetCache
from qlab.features import simple_returns, realized_volatility, zscore, winsorize
from qlab.alphas import momentum, mean_reversion_zscore, low_volatility
from qlab.portfolio import equal_weight_long_short, normalize_weights, apply_position_limits
from qlab.backtest import run_backtest, BacktestConfig
from qlab.risk import performance_summary, drawdown_series

# Fetch data
provider = ParquetCache(YFinanceProvider(), cache_dir=".qlab_cache")
prices = provider.fetch(["AAPL", "MSFT", "GOOG"], "2022-01-01", "2026-01-01")
close = prices["adj_close"]

# Compute alpha signal
signal = momentum(close, lookback=252, skip=21)

# Construct portfolio
weights = equal_weight_long_short(signal.dropna(), long_pct=0.2, short_pct=0.2)
weights = normalize_weights(weights, gross_exposure=2.0, net_exposure=0.0)

# Backtest
config = BacktestConfig(
    rebalance_freq="monthly",
    commission_bps=5,
    slippage_bps=5,
    signal_lag=1,
    execution_price="open",
)
result = run_backtest(weights, prices, config=config)

# Analyze
summary = performance_summary(result.portfolio_returns)
for k, v in summary.items():
    print(f"{k:25s}: {v:>10.4f}")
```

### B) Backend / Research Engine

**Run stock selection via CLI:**

```bash
# Default: 20 long-only stocks from S&P 500
python -m app.run_selection

# Custom parameters
python -m app.run_selection --n-long 30 --rebalance-freq monthly --max-position 0.05

# Long-short mode
python -m app.run_selection --long-short --n-short 15

# Skip regime filter
python -m app.run_selection --no-regime-filter
```

**Outputs** are saved to `runs/<run_id>/`:

```
runs/20260208_091503_015758/
├── config.json              Pipeline configuration
├── selection.csv            Selected stocks with scores, weights, sectors
├── signal_details.csv       Per-ticker signal values for all alphas
├── backtest_summary.json    Sharpe, return, drawdown, hit rate
├── filter_log.json          Which tickers were filtered and why
├── regime.json              Market regime classification
└── plots/
    ├── equity_curve.png     Cumulative return and drawdown
    ├── turnover.png         Portfolio turnover over time
    └── selection_scores.png Horizontal bar chart of selected stocks
```

**Run via FastAPI backend:**

```bash
# Start the backend server
uvicorn gui.backend.main:app --host 0.0.0.0 --port 8765

# API endpoints:
# POST /api/selection/run          — Run a new stock selection
# GET  /api/selection/runs         — List all completed runs
# GET  /api/selection/runs/{id}/status    — Check run status
# GET  /api/selection/runs/{id}/selection — Get selected stocks
# GET  /api/selection/runs/{id}/report    — Full run report (config, backtest, regime)
# GET  /api/health                 — Health check
```

### C) Notebooks

Six tutorials in `docs/tutorials/` demonstrate end-to-end workflows using real yfinance data:

| # | Notebook | Topic |
|---|----------|-------|
| 01 | `01_cross_sectional_momentum.ipynb` | Momentum alpha, IC analysis, quintile spreads |
| 02 | `02_mean_reversion.ipynb` | Z-score mean reversion, Bollinger bands |
| 03 | `03_volatility_alphas.ipynb` | Low-vol anomaly, vol targeting, GARCH basics |
| 04 | `04_multi_factor_composition.ipynb` | IC-weighted factor combination, sector neutralization |
| 05 | `05_extension_points.ipynb` | Custom alphas, data providers, constraint functions |
| 06 | `06_news_to_signals.ipynb` | News sentiment via FinBERT, event study framework |

**Running notebooks:**

```bash
jupyter lab docs/tutorials/
```

Notebooks import directly from the `qlab` package and produce inline matplotlib visualizations. Each notebook is self-contained and runs without external dependencies beyond the core install.

### D) GUI

The GUI provides an interactive dark-themed web interface for strategy development.

**Launch in production mode (single port):**

```bash
# Build frontend
cd gui/frontend && npm run build && cd ../..

# Serve everything from FastAPI
uvicorn gui.backend.main:app --host 0.0.0.0 --port 8765
```

**Launch in development mode (hot reload):**

```bash
bash gui/run.sh
```

**GUI features:**

- **Dashboard** — View backtest results, equity curves, drawdowns, rolling Sharpe
- **Strategy Builder** — Configure and launch backtests with custom parameters
- **Alpha Lab** — Analyze individual alpha signals with IC series and quantile returns

**Limitations:** The GUI is designed for visual exploration. For production research, parameter sweeps, and batch processing, use the Python library or CLI directly.

## Data Workflow

**Expected formats:**

All data flows through stacked `MultiIndex(date, ticker)` DataFrames:

| Column | Description |
|--------|-------------|
| `open`, `high`, `low`, `close` | Raw OHLC prices |
| `adj_close` | Split/dividend-adjusted close (primary signal input) |
| `volume` | Daily share volume |

**Data providers:**

- `YFinanceProvider` — Fetches from Yahoo Finance API (rate-limited)
- `ParquetCache` — Wraps any provider with local Parquet disk cache
- `CsvProvider` — Load from local CSV files

**Caching:**

```python
provider = ParquetCache(YFinanceProvider(), cache_dir=".qlab_cache")
# First call fetches from API; subsequent calls read from disk
```

**Reproducibility notes:**

- All pipeline runs save their full config as `config.json`
- Data is cached locally to ensure identical inputs across runs
- Random seeds are fixed where applicable

**Bias awareness:**

- **Survivorship bias** — The provided `universe.csv` reflects current S&P 500 membership. For rigorous research, use point-in-time constituent lists.
- **Lookahead bias** — Alpha signals use only data available up to each date. The `signal_lag` parameter in `BacktestConfig` (default: 1 day) prevents same-day execution. Unit tests verify no-lookahead properties.

## Alpha Research Philosophy

**Cross-sectional focus.** Most alphas in this platform rank stocks relative to peers on each date, not in absolute terms. This makes signals more robust to market regimes.

**Universe and constraints matter.** A signal that looks good unconstrained may fail under realistic sector caps, position limits, and liquidity filters. Always test with constraints enabled.

**Transaction costs and turnover.** Every backtest includes commission and slippage modeling. High-turnover strategies degrade under realistic costs. Monitor turnover per rebalance.

**Diagnostics over raw performance.** Focus on:
- Information Coefficient (IC) and its stability
- Quintile spread monotonicity
- Hit rate and profit factor
- Drawdown duration and depth

A strategy with Sharpe 1.0 and stable IC is more trustworthy than one with Sharpe 2.0 driven by a few outlier months.

## Extending the Platform

### Adding a new alpha

Create a function in `qlab/alphas/` that accepts a `pd.Series` (adj_close with `MultiIndex(date, ticker)`) and returns a `pd.Series` with the same index:

```python
# qlab/alphas/custom.py
def my_alpha(close: pd.Series, lookback: int = 60) -> pd.Series:
    returns = close.groupby(level="ticker").pct_change(lookback)
    return returns.groupby(level="date").apply(lambda x: (x - x.mean()) / x.std())
```

Register it in `qlab/alphas/__init__.py` and add to `SelectionConfig.alpha_weights`.

### Adding a new data provider

Implement the `DataProvider` protocol:

```python
class MyProvider:
    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        # Return stacked OHLCV DataFrame with MultiIndex(date, ticker)
        ...
```

Wrap with `ParquetCache` for disk caching.

### Adding ML models

Notebooks 05 and 06 demonstrate the pattern:
1. Compute features using `qlab.features`
2. Train model on historical cross-sections
3. Generate predictions as a `pd.Series` with `MultiIndex(date, ticker)`
4. Feed into `qlab.portfolio` and `qlab.backtest` as usual

### Adding new visualizations

The GUI frontend uses Recharts. Add new chart components in `gui/frontend/src/components/Chart.tsx` and wire them into the relevant page components.

## Running Tests

```bash
# All tests
pytest

# Core library tests only
pytest tests/ -v

# StockPicker pipeline tests
pytest tests/test_stock_picker.py -v

# GUI backend API tests
pytest gui/backend/tests/test_api.py -v
```

## Disclaimer

This platform is for **research and educational purposes only**. It does not constitute investment advice, a recommendation to buy or sell any security, or an offer to provide investment management services.

- Backtested results do not guarantee future performance.
- The user is solely responsible for validating strategies, managing risk, and ensuring compliance with applicable regulations.
- No warranty is provided regarding the accuracy or completeness of data, signals, or analytics.

Use at your own risk.
