# Design notes

## Data representation

The library uses a **stacked (long-form) DataFrame** with a two-level
`MultiIndex(date, ticker)` as its canonical data format.

**Why this choice:**

| Alternative            | Limitation                                                 |
|------------------------|------------------------------------------------------------|
| Wide DataFrame         | Requires all tickers on every date; awkward for ragged panels. |
| `pd.Panel` (deprecated)| Removed in pandas 1.0.                                    |
| `xarray.DataArray`     | Adds a dependency; less natural for groupby operations.    |
| Dict of DataFrames     | No automatic alignment; easy to introduce mismatched dates.|

The stacked layout supports both operation modes naturally:

- **Time-series** (per ticker): `series.groupby(level="ticker").transform(...)`
- **Cross-sectional** (per date): `series.groupby(level="date").transform(...)`

When a matrix view is needed (e.g. for the backtest engine), `unstack(level="ticker")`
converts to a date × ticker wide DataFrame.  This conversion is localised inside
functions and not exposed to the user.

## Timing convention

The default backtest timing is:

1. **Signal observation**: close of day *t*.
2. **Trade execution**: open of day *t + signal_lag* (default `signal_lag=1`).
3. **Return accrual**: open-to-close on the execution day, then close-to-close on
   subsequent days until the next rebalance.

This is conservative — it prevents look-ahead bias (you cannot trade on data you
observe at the close until the next session opens) while still allowing same-day
participation in returns from the execution day.

## Transaction costs

The cost model is intentionally simple:

```
cost = |trade_value| × (commission_bps + slippage_bps) / 10000
```

This is a **linear** model.  It does not capture:

- Non-linear market impact (Almgren-Chriss, square-root law)
- Bid-ask spread variation by liquidity
- Short-borrowing costs

These are appropriate extensions for production use but would add complexity
beyond the scope of a research library.

## Alpha functions

Every alpha is a **pure function**:

```python
def alpha_name(prices: pd.Series, ...) -> pd.Series:
    ...
```

No hidden state, no side effects, no class instances.  This makes alphas:

- Easy to test in isolation.
- Safe to combine (they share no mutable state).
- Trivially parallelisable across tickers or time periods.

## Portfolio construction

Weight construction is separated from signal generation:

```
signal → weights → constraints → backtest
```

This pipeline makes each stage independently testable and swappable.

## Assumptions

- **Universe is fixed**: the library does not handle index reconstitution,
  IPOs, delistings, or survivorship bias.  Extend by filtering the price
  DataFrame before passing it in.
- **No intraday data**: all operations assume daily bars.
- **Single currency**: no FX conversion.
- **Fully invested**: the backtest does not model cash drag from partial
  fills or margin requirements.

## Extension points

| Area                   | How to extend                                              |
|------------------------|------------------------------------------------------------|
| New data source        | Subclass `MarketDataProvider`, implement `fetch()`.        |
| New alpha              | Add a pure function to `qlab/alphas/`.                     |
| Non-linear costs       | Replace or subclass the cost model in `qlab/backtest/costs.py`. |
| Multi-asset            | Generalise the ticker dimension to include asset class.    |
| Risk model             | Add a covariance estimator in `qlab/risk/`.                |
| Optimisation           | Add a mean-variance or risk-parity module in `qlab/portfolio/`. |
| Execution simulation   | Replace the fill model to use volume participation limits. |

## Reproducibility

- All random operations require an explicit `seed` parameter.
- No global mutable state.
- Configuration is via frozen `dataclass` instances.
- `numpy.random.default_rng` is preferred over legacy `numpy.random.*`.
