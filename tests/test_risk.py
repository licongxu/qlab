"""Tests for qlab.risk."""

import numpy as np
import pandas as pd
import pytest

from qlab.risk.metrics import (
    total_return,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    hit_rate,
    profit_factor,
    performance_summary,
)
from qlab.risk.drawdown import drawdown_series, drawdown_details
from qlab.risk.regression import factor_regression


@pytest.fixture()
def daily_returns():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=500)
    return pd.Series(rng.normal(0.0003, 0.01, 500), index=dates)


class TestMetrics:
    def test_total_return_simple(self):
        r = pd.Series([0.01, 0.02, -0.01])
        expected = (1.01 * 1.02 * 0.99) - 1
        np.testing.assert_allclose(total_return(r), expected, rtol=1e-10)

    def test_annualized_return_sign(self, daily_returns):
        # With positive drift, annualized return should be positive
        assert annualized_return(daily_returns) > 0

    def test_annualized_vol_positive(self, daily_returns):
        assert annualized_volatility(daily_returns) > 0

    def test_sharpe_positive_for_positive_drift(self, daily_returns):
        assert sharpe_ratio(daily_returns) > 0

    def test_max_drawdown_negative(self, daily_returns):
        assert max_drawdown(daily_returns) <= 0

    def test_hit_rate_bounded(self, daily_returns):
        hr = hit_rate(daily_returns)
        assert 0 <= hr <= 1

    def test_profit_factor_positive(self, daily_returns):
        pf = profit_factor(daily_returns)
        assert pf > 0

    def test_performance_summary_keys(self, daily_returns):
        summary = performance_summary(daily_returns)
        expected_keys = {
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "hit_rate", "profit_factor", "num_days",
        }
        assert set(summary.keys()) == expected_keys


class TestDrawdown:
    def test_drawdown_series_at_peak_is_zero(self, daily_returns):
        dd = drawdown_series(daily_returns)
        assert dd.max() <= 1e-10

    def test_drawdown_series_non_positive(self, daily_returns):
        dd = drawdown_series(daily_returns)
        assert (dd <= 1e-10).all()

    def test_drawdown_details_columns(self, daily_returns):
        details = drawdown_details(daily_returns)
        expected_cols = {"start", "trough", "end", "depth", "days", "recovery_days"}
        assert expected_cols == set(details.columns)

    def test_drawdown_details_sorted_by_depth(self, daily_returns):
        details = drawdown_details(daily_returns)
        if len(details) > 1:
            assert details["depth"].iloc[0] <= details["depth"].iloc[-1]


class TestRegression:
    def test_single_factor_regression(self, daily_returns):
        rng = np.random.default_rng(99)
        # Construct a factor that explains some of the returns
        mkt = pd.Series(
            rng.normal(0.0003, 0.012, len(daily_returns)),
            index=daily_returns.index,
        )
        port = 0.5 * mkt + pd.Series(
            rng.normal(0, 0.005, len(daily_returns)),
            index=daily_returns.index,
        )
        factors = pd.DataFrame({"market": mkt})
        result = factor_regression(port, factors)
        assert "market" in result.betas
        # Beta should be close to 0.5
        np.testing.assert_allclose(result.betas["market"], 0.5, atol=0.1)
        assert 0 <= result.r_squared <= 1
