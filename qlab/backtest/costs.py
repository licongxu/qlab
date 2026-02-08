"""Transaction cost and slippage model."""

from __future__ import annotations

import pandas as pd

from qlab.backtest.config import BacktestConfig


def transaction_costs(
    trades: pd.Series,
    prices: pd.Series,
    config: BacktestConfig,
) -> pd.Series:
    """Compute total transaction costs for a set of trades.

    Parameters
    ----------
    trades : Series
        Signed dollar trade amounts (positive = buy), MultiIndex ``(date, ticker)``.
    prices : Series
        Execution prices at the time of the trade, same index as *trades*.
    config : BacktestConfig
        Contains commission and slippage parameters.

    Returns
    -------
    Series
        Per-position cost in dollars (always non-negative), same index.
    """
    abs_traded = trades.abs()
    commission = abs_traded * (config.commission_bps / 10_000)
    slippage = abs_traded * (config.slippage_bps / 10_000)
    return commission + slippage
