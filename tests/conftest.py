"""Shared test fixtures for qlab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_dates() -> pd.DatetimeIndex:
    """100 business days starting 2020-01-02."""
    return pd.bdate_range("2020-01-02", periods=100, freq="B")


@pytest.fixture()
def tickers() -> list[str]:
    return ["AAPL", "MSFT", "GOOG", "AMZN", "META"]


@pytest.fixture()
def sample_prices(sample_dates, tickers) -> pd.DataFrame:
    """Deterministic synthetic stacked OHLCV data."""
    rng = np.random.default_rng(0)
    parts = []
    for ticker in tickers:
        n = len(sample_dates)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
        high = close * (1 + rng.uniform(0, 0.02, n))
        low = close * (1 - rng.uniform(0, 0.02, n))
        open_ = close * np.exp(rng.normal(0, 0.005, n))
        volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "adj_close": close,
            },
            index=pd.MultiIndex.from_arrays(
                [sample_dates, np.full(n, ticker)], names=["date", "ticker"]
            ),
        )
        parts.append(df)
    return pd.concat(parts).sort_index()


@pytest.fixture()
def close_series(sample_prices) -> pd.Series:
    """Adj close as a Series."""
    return sample_prices["adj_close"]


@pytest.fixture()
def sample_returns(close_series) -> pd.Series:
    """Simple daily returns."""
    return close_series.groupby(level="ticker").pct_change().dropna()


@pytest.fixture()
def sample_signal(sample_prices, tickers, sample_dates) -> pd.Series:
    """Deterministic signal: rank of close price (cross-sectional)."""
    rng = np.random.default_rng(1)
    idx = sample_prices.index
    return pd.Series(rng.standard_normal(len(idx)), index=idx, name="signal")


@pytest.fixture()
def sample_weights(sample_signal) -> pd.Series:
    """Simple equal-weight long-short from sample signal."""
    from qlab.portfolio.construction import equal_weight_long_short
    return equal_weight_long_short(sample_signal, long_pct=0.4, short_pct=0.4)
