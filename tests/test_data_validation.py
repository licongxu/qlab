"""Tests for data integrity validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.utils.validation import validate_market_data


def _make_good_data():
    """Create valid synthetic market data."""
    dates = pd.bdate_range("2020-01-02", periods=50, freq="B")
    tickers = ["AAPL", "MSFT"]
    parts = []
    for t in tickers:
        n = len(dates)
        close = np.linspace(100, 120, n)
        df = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(n, 1e6),
                "adj_close": close,
            },
            index=pd.MultiIndex.from_arrays(
                [dates, np.full(n, t)], names=["date", "ticker"]
            ),
        )
        parts.append(df)
    return pd.concat(parts).sort_index()


class TestValidateMarketData:
    def test_good_data_passes(self):
        prices = _make_good_data()
        result = validate_market_data(prices)
        assert result["valid"] is True
        assert result["global_issues"] == []
        assert result["ticker_issues"] == {}

    def test_duplicate_index(self):
        prices = _make_good_data()
        dup = pd.concat([prices, prices.iloc[:2]])
        result = validate_market_data(dup)
        assert result["valid"] is False
        assert any("duplicate" in i for i in result["global_issues"])

    def test_negative_close(self):
        prices = _make_good_data()
        prices.iloc[0, prices.columns.get_loc("close")] = -1.0
        result = validate_market_data(prices)
        assert result["valid"] is False
        assert any("non-positive" in iss for issues in result["ticker_issues"].values() for iss in issues)

    def test_negative_volume(self):
        prices = _make_good_data()
        prices.iloc[0, prices.columns.get_loc("volume")] = -100.0
        result = validate_market_data(prices)
        assert result["valid"] is False
        assert any("negative volume" in iss for issues in result["ticker_issues"].values() for iss in issues)

    def test_high_missing_rate(self):
        prices = _make_good_data()
        # Set 20% of closes to NaN for one ticker
        aapl_mask = prices.index.get_level_values("ticker") == "AAPL"
        n_aapl = aapl_mask.sum()
        nan_indices = prices.index[aapl_mask][:int(n_aapl * 0.2)]
        prices.loc[nan_indices, "close"] = np.nan
        result = validate_market_data(prices, max_missing_rate=0.05)
        assert "AAPL" in result["ticker_issues"]
        assert any("missing rate" in i for i in result["ticker_issues"]["AAPL"])
