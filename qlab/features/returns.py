"""Return calculations on stacked price data.

All functions accept a stacked ``adj_close`` Series (or DataFrame column)
with MultiIndex ``(date, ticker)`` and return a Series in the same layout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Arithmetic period returns: ``(P_t / P_{t-periods}) - 1``.

    Parameters
    ----------
    prices : Series
        Adjusted close prices with MultiIndex ``(date, ticker)``.
    periods : int
        Lag in trading days.

    Returns
    -------
    Series
        Simple returns, NaN for the first *periods* observations per ticker.
    """
    return prices.groupby(level="ticker").pct_change(periods=periods)


def log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Logarithmic period returns: ``ln(P_t / P_{t-periods})``.

    Parameters
    ----------
    prices : Series
        Adjusted close prices.
    periods : int
        Lag in trading days.
    """
    return np.log(prices / prices.groupby(level="ticker").shift(periods))


def cumulative_returns(returns: pd.Series, periods: int = 21) -> pd.Series:
    """Rolling cumulative simple return over *periods* days.

    Uses the product of ``(1 + r)`` over the window minus one.
    """
    def _cum(s: pd.Series) -> pd.Series:
        return (1 + s).rolling(window=periods, min_periods=periods).apply(
            np.prod, raw=True
        ) - 1

    return returns.groupby(level="ticker").apply(_cum, include_groups=False).droplevel(0)


def excess_returns(
    returns: pd.Series,
    benchmark: pd.Series,
) -> pd.Series:
    """Returns in excess of a benchmark.

    Parameters
    ----------
    returns : Series
        Asset returns with MultiIndex ``(date, ticker)``.
    benchmark : Series
        Benchmark return indexed by date only (one value per day).

    Returns
    -------
    Series
        ``returns - benchmark`` after aligning on the date level.
    """
    dates = returns.index.get_level_values("date")
    bench_aligned = benchmark.reindex(dates).values
    return returns - bench_aligned
