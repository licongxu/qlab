"""Regression tests for equity curve timing (Issue A).

The equity curve should start moving from the correct first execution day,
not include a long flat warmup period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlab.backtest.config import BacktestConfig
from qlab.backtest.engine import run_backtest


def _make_synthetic_data(n_dates=300, n_tickers=5, seed=42):
    """Create synthetic prices and weights with a known warmup gap.

    Weights are zero for the first ``warmup`` dates and non-zero after.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]

    parts = []
    for t in tickers:
        n = len(dates)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
        open_ = close * np.exp(rng.normal(0, 0.005, n))
        df = pd.DataFrame(
            {"open": open_, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": 1e6, "adj_close": close},
            index=pd.MultiIndex.from_arrays(
                [dates, np.full(n, t)], names=["date", "ticker"]
            ),
        )
        parts.append(df)
    prices = pd.concat(parts).sort_index()

    # Weights: zero for first 200 dates, then equal weight
    warmup = 200
    weight_dates = dates[warmup:]
    weight_parts = []
    for t in tickers:
        w = pd.Series(
            1.0 / n_tickers,
            index=pd.MultiIndex.from_arrays(
                [weight_dates, np.full(len(weight_dates), t)],
                names=["date", "ticker"],
            ),
        )
        weight_parts.append(w)
    weights = pd.concat(weight_parts).sort_index()

    return prices, weights, dates, warmup


class TestEquityCurveTiming:
    """Equity curve should start from first execution day, not beginning of data."""

    def test_result_trimmed_to_first_execution(self):
        """Portfolio returns should not include the long flat warmup period."""
        prices, weights, dates, warmup = _make_synthetic_data()
        cfg = BacktestConfig(
            rebalance_freq="daily", signal_lag=1,
            commission_bps=0, slippage_bps=0,
        )
        result = run_backtest(weights, prices, config=cfg)

        # The result should start near the first weight date + signal_lag,
        # NOT from the beginning of the price data.
        first_ret_date = result.portfolio_returns.index[0]
        first_weight_date = dates[warmup]
        # Allow 1 day before first execution for baseline
        assert first_ret_date >= dates[warmup - 1], (
            f"Result starts at {first_ret_date}, expected near {first_weight_date}"
        )
        # Definitely should NOT start at the beginning of prices
        assert first_ret_date > dates[10], (
            f"Result starts at {first_ret_date}, too early — warmup not trimmed"
        )

    def test_first_nonzero_return_is_execution_day(self):
        """The first non-zero return should be on the execution day (warmup + lag)."""
        prices, weights, dates, warmup = _make_synthetic_data()
        cfg = BacktestConfig(
            rebalance_freq="daily", signal_lag=1,
            commission_bps=0, slippage_bps=0,
        )
        result = run_backtest(weights, prices, config=cfg)

        nonzero = result.portfolio_returns[result.portfolio_returns.abs() > 1e-12]
        assert len(nonzero) > 0, "No non-zero returns found"
        first_nonzero = nonzero.index[0]
        # With signal_lag=1, first execution is warmup_date + 1
        expected = dates[warmup + 1]
        assert first_nonzero == expected, (
            f"First non-zero return at {first_nonzero}, expected {expected}"
        )

    def test_positions_nonzero_on_execution_day(self):
        """Positions should be non-zero starting from the execution day."""
        prices, weights, dates, warmup = _make_synthetic_data()
        cfg = BacktestConfig(
            rebalance_freq="daily", signal_lag=1,
            commission_bps=0, slippage_bps=0,
        )
        result = run_backtest(weights, prices, config=cfg)

        pos_sum = result.positions.abs().sum(axis=1)
        first_pos = pos_sum[pos_sum > 1e-10].index[0]
        expected = dates[warmup + 1]
        assert first_pos == expected

    def test_no_flat_prefix_in_equity(self):
        """Cumulative return should not start with a long flat region."""
        prices, weights, dates, warmup = _make_synthetic_data()
        cfg = BacktestConfig(
            rebalance_freq="daily", signal_lag=1,
            commission_bps=0, slippage_bps=0,
        )
        result = run_backtest(weights, prices, config=cfg)
        cum_ret = (1 + result.portfolio_returns).cumprod()

        # Within 3 days of the start, the equity curve should have moved
        n_flat = (cum_ret.iloc[:3] == 1.0).sum()
        # At most 2 flat days (baseline + first execution day where ret could be ~0)
        assert n_flat <= 2, f"Too many flat days at start: {n_flat}"

    def test_zero_weight_preserved_after_rebalance(self):
        """A ticker with weight=0 on a signal date should not carry stale weights."""
        dates = pd.bdate_range("2020-01-02", periods=50, freq="B")
        tickers = ["A", "B"]

        parts = []
        for t in tickers:
            close = pd.Series(
                100.0, index=pd.MultiIndex.from_arrays(
                    [dates, np.full(len(dates), t)], names=["date", "ticker"]
                )
            )
            parts.append(close)
        close_s = pd.concat(parts).sort_index()
        prices = pd.DataFrame({
            "open": close_s, "high": close_s, "low": close_s,
            "close": close_s, "volume": 1e6, "adj_close": close_s,
        })

        # Signal date 1: A=0.5, B=0.5
        # Signal date 2: A=0.0, B=1.0 (A should go to zero)
        d1, d2 = dates[5], dates[15]
        idx_a1 = (d1, "A")
        idx_b1 = (d1, "B")
        idx_a2 = (d2, "A")
        idx_b2 = (d2, "B")

        weights = pd.Series(
            [0.5, 0.5, 0.0, 1.0],
            index=pd.MultiIndex.from_tuples(
                [idx_a1, idx_b1, idx_a2, idx_b2], names=["date", "ticker"]
            ),
        )

        cfg = BacktestConfig(
            rebalance_freq="daily", signal_lag=1,
            commission_bps=0, slippage_bps=0,
            execution_price="close",
        )
        result = run_backtest(weights, prices, config=cfg)

        # After d2 + lag, ticker A should have zero weight
        exec_date = dates[16]  # d2 + 1
        if exec_date in result.positions.index:
            a_weight = result.positions.loc[exec_date, "A"]
            assert abs(a_weight) < 1e-10, (
                f"Ticker A should be 0 after rebalance, got {a_weight}"
            )

    def test_open_execution_uses_open_to_close_return(self):
        """When execution_price='open', rebalance-day return should use open-to-close."""
        dates = pd.bdate_range("2020-01-02", periods=20, freq="B")
        tickers = ["X"]

        # Construct prices where open != close for a clear test
        n = len(dates)
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        # On execution day (day 6 = signal day 5 + lag 1), set distinct values
        open_[6] = 95.0
        close[6] = 105.0  # open-to-close return = (105-95)/95 = 10.526%
        # close-to-close = (105-100)/100 = 5%

        idx = pd.MultiIndex.from_arrays(
            [dates, np.full(n, "X")], names=["date", "ticker"]
        )
        prices = pd.DataFrame(
            {"open": open_, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": 1e6, "adj_close": close},
            index=idx,
        )

        weights = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples(
                [(dates[5], "X")], names=["date", "ticker"]
            ),
        )

        cfg = BacktestConfig(
            rebalance_freq="daily", signal_lag=1,
            commission_bps=0, slippage_bps=0,
            execution_price="open",
        )
        result = run_backtest(weights, prices, config=cfg)

        # Find the execution day return
        exec_day = dates[6]
        if exec_day in result.portfolio_returns.index:
            ret = result.portfolio_returns.loc[exec_day]
            expected = (105.0 - 95.0) / 95.0  # open-to-close
            assert abs(ret - expected) < 1e-10, (
                f"Expected open-to-close return {expected:.4f}, got {ret:.4f}"
            )
