"""Tests for qlab.backtest."""

import numpy as np
import pandas as pd
import pytest

from qlab.backtest.config import BacktestConfig
from qlab.backtest.engine import run_backtest, BacktestResult


class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 1_000_000.0
        assert cfg.commission_bps == 5.0
        assert cfg.rebalance_freq == "monthly"
        assert cfg.signal_lag == 1
        assert cfg.execution_price == "open"

    def test_frozen(self):
        cfg = BacktestConfig()
        with pytest.raises(AttributeError):
            cfg.initial_capital = 0


class TestBacktestEngine:
    def test_runs_without_error(self, sample_weights, sample_prices):
        cfg = BacktestConfig(
            rebalance_freq="daily",
            commission_bps=0,
            slippage_bps=0,
        )
        result = run_backtest(sample_weights, sample_prices, config=cfg)
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_returns) > 0

    def test_zero_cost_gross_equals_net(self, sample_weights, sample_prices):
        cfg = BacktestConfig(
            rebalance_freq="daily",
            commission_bps=0,
            slippage_bps=0,
        )
        result = run_backtest(sample_weights, sample_prices, config=cfg)
        np.testing.assert_allclose(
            result.portfolio_returns.values,
            result.gross_returns.values,
            atol=1e-12,
        )

    def test_costs_reduce_returns(self, sample_weights, sample_prices):
        cfg_no_cost = BacktestConfig(
            rebalance_freq="daily", commission_bps=0, slippage_bps=0
        )
        cfg_cost = BacktestConfig(
            rebalance_freq="daily", commission_bps=10, slippage_bps=10
        )
        r0 = run_backtest(sample_weights, sample_prices, config=cfg_no_cost)
        r1 = run_backtest(sample_weights, sample_prices, config=cfg_cost)
        # Cumulative return with costs should be <= without costs
        cum0 = (1 + r0.portfolio_returns).prod()
        cum1 = (1 + r1.portfolio_returns).prod()
        assert cum1 <= cum0 + 1e-10

    def test_positions_shape(self, sample_weights, sample_prices):
        cfg = BacktestConfig(rebalance_freq="daily")
        result = run_backtest(sample_weights, sample_prices, config=cfg)
        assert result.positions.shape[0] > 0
        assert result.positions.shape[1] > 0

    def test_monthly_rebalance_less_turnover(self, sample_weights, sample_prices):
        cfg_daily = BacktestConfig(rebalance_freq="daily", commission_bps=0, slippage_bps=0)
        cfg_monthly = BacktestConfig(rebalance_freq="monthly", commission_bps=0, slippage_bps=0)
        r_daily = run_backtest(sample_weights, sample_prices, config=cfg_daily)
        r_monthly = run_backtest(sample_weights, sample_prices, config=cfg_monthly)
        assert r_monthly.turnover.sum() <= r_daily.turnover.sum() + 1e-10
