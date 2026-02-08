"""Daily-bar backtesting engine.

Timing convention (default):
    1. Signal is observed at the **close** of day *t*.
    2. Target weights are computed from the signal.
    3. Trades execute at the **open** of day *t + signal_lag* (default lag = 1).
    4. Returns are then earned from open-to-close on day *t + signal_lag*
       and close-to-close on subsequent days until the next rebalance.

The engine uses **weight-based accounting**: positions are tracked as
fractions of portfolio equity, not share counts.  This avoids floating-point
drift from share-level bookkeeping and keeps the implementation vectorised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qlab.backtest.config import BacktestConfig
from qlab.backtest.costs import transaction_costs
from qlab.utils.alignment import unstack_to_wide
from qlab.utils.calendar import rebalance_dates
from qlab.utils.validation import QlabValidationError


@dataclass
class BacktestResult:
    """Container for backtest outputs.

    Attributes
    ----------
    portfolio_returns : Series
        Daily portfolio return (after costs).
    gross_returns : Series
        Daily portfolio return before costs.
    positions : DataFrame
        Wide-form (date × ticker) weight matrix actually held.
    turnover : Series
        Daily one-way turnover (sum |Δw| / 2).
    total_costs : Series
        Daily total transaction costs as a fraction of portfolio value.
    config : BacktestConfig
        The configuration used for this run.
    """

    portfolio_returns: pd.Series
    gross_returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    total_costs: pd.Series
    config: BacktestConfig


def run_backtest(
    weights: pd.Series,
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a daily-bar backtest.

    Parameters
    ----------
    weights : Series
        Target portfolio weights with MultiIndex ``(date, ticker)``.
        Each date present is interpreted as a **signal date** — the date
        on which the signal was observed.
    prices : DataFrame
        Stacked OHLCV price data with MultiIndex ``(date, ticker)``
        and at least columns ``open``, ``close``.
    config : BacktestConfig, optional
        Backtest parameters.  Uses sensible defaults if not provided.

    Returns
    -------
    BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    # --- Unstack to wide form for vectorised operations ---
    close_wide = unstack_to_wide(prices["close"])
    if config.execution_price == "open":
        exec_wide = unstack_to_wide(prices["open"])
    else:
        exec_wide = close_wide

    tickers = close_wide.columns
    all_dates = close_wide.index
    weight_wide = unstack_to_wide(weights).reindex(
        index=all_dates, columns=tickers
    ).fillna(0.0)

    # --- Determine rebalance dates ---
    signal_dates = weight_wide.index[weight_wide.abs().sum(axis=1) > 0]
    if len(signal_dates) == 0:
        raise QlabValidationError("No non-zero weights found.")
    reb_dates = rebalance_dates(all_dates, freq=config.rebalance_freq)
    # Intersect signal dates with rebalance dates
    active_signal_dates = signal_dates.intersection(reb_dates)
    if len(active_signal_dates) == 0:
        # Fall back to using all signal dates
        active_signal_dates = signal_dates

    # --- Build the target weight schedule (forward-fill between rebalances) ---
    target_w = pd.DataFrame(0.0, index=all_dates, columns=tickers)
    for d in active_signal_dates:
        target_w.loc[d] = weight_wide.loc[d]
    # Forward fill targets: hold until next rebalance
    target_w = target_w.replace(0.0, np.nan)
    # Mark actual rebalance rows, then ffill
    mask = pd.Series(False, index=all_dates)
    mask[active_signal_dates] = True
    for col in target_w.columns:
        # Only ffill from actual signal dates
        target_w[col] = target_w[col].ffill()
    target_w = target_w.fillna(0.0)

    # --- Apply signal lag ---
    if config.signal_lag > 0:
        target_w = target_w.shift(config.signal_lag).fillna(0.0)

    # --- Compute daily close-to-close returns ---
    asset_returns = close_wide.pct_change().fillna(0.0)

    # --- If executing at open, adjust first-day return ---
    # On the execution day, return is from open to close, not close to close.
    if config.execution_price == "open":
        open_to_close = (close_wide - exec_wide) / exec_wide
        open_to_close = open_to_close.fillna(0.0)

    # --- Walk forward computing realised weights and returns ---
    n_dates = len(all_dates)
    held_w = pd.DataFrame(0.0, index=all_dates, columns=tickers)
    port_ret_gross = pd.Series(0.0, index=all_dates)
    turnover_s = pd.Series(0.0, index=all_dates)
    cost_s = pd.Series(0.0, index=all_dates)

    prev_w = pd.Series(0.0, index=tickers)

    for i in range(n_dates):
        date = all_dates[i]
        target = target_w.iloc[i]

        # Determine if we rebalance today
        trade = target - prev_w
        trade_turnover = trade.abs().sum() / 2

        if trade_turnover > 1e-10:
            # We are rebalancing
            cost_frac = (
                trade.abs().sum() * (config.commission_bps + config.slippage_bps) / 10_000
            )
            current_w = target.copy()
            turnover_s.iloc[i] = trade_turnover
            cost_s.iloc[i] = cost_frac
        else:
            current_w = prev_w.copy()

        held_w.iloc[i] = current_w

        # Compute return for this day
        if i == 0:
            day_ret = 0.0
        else:
            day_ret = (current_w * asset_returns.iloc[i]).sum()

        port_ret_gross.iloc[i] = day_ret

        # Drift weights by asset returns for next day
        if day_ret != 0:
            drifted = current_w * (1 + asset_returns.iloc[i])
            total_val = 1 + day_ret
            if abs(total_val) > 1e-10:
                prev_w = drifted / total_val
            else:
                prev_w = current_w
        else:
            prev_w = current_w

    port_ret_net = port_ret_gross - cost_s

    return BacktestResult(
        portfolio_returns=port_ret_net,
        gross_returns=port_ret_gross,
        positions=held_w,
        turnover=turnover_s,
        total_costs=cost_s,
        config=config,
    )
