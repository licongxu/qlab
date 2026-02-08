"""Momentum alpha signals.

Each function is a pure mapping from price/return data to a signal Series.
Convention: higher signal → more attractive (long).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.features.returns import simple_returns, log_returns
from qlab.features.cross_section import rank


def momentum(
    prices: pd.Series,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """Classic cross-sectional momentum (Jegadeesh & Titman, 1993).

    Computes the cumulative return over ``[t-lookback, t-skip]``, skipping the
    most recent *skip* days to avoid the short-term reversal effect.

    Parameters
    ----------
    prices : Series
        Adjusted close prices, MultiIndex ``(date, ticker)``.
    lookback : int
        Total lookback window (default 252 ≈ 12 months).
    skip : int
        Recent days to skip (default 21 ≈ 1 month).

    Returns
    -------
    Series
        Raw momentum signal (cumulative return).
    """
    if skip >= lookback:
        raise ValueError("skip must be less than lookback")
    full_ret = simple_returns(prices, periods=lookback)
    skip_ret = simple_returns(prices, periods=skip)
    # (1+r_full)/(1+r_skip) - 1
    signal = (1 + full_ret) / (1 + skip_ret) - 1
    return signal


def short_term_reversal(
    prices: pd.Series,
    lookback: int = 21,
) -> pd.Series:
    """Short-term reversal: negative of recent return.

    Stocks that fell recently are expected to bounce back.

    Parameters
    ----------
    prices : Series
        Adjusted close prices.
    lookback : int
        Window in trading days (default 21 ≈ 1 month).
    """
    ret = simple_returns(prices, periods=lookback)
    return -ret


def trend_strength(
    prices: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """Trend strength measured as the t-statistic of daily returns.

    A stock with consistently positive returns will have a high t-stat
    even if the cumulative return is moderate.  This penalises noisy winners.

    Parameters
    ----------
    prices : Series
        Adjusted close prices.
    lookback : int
        Window in trading days.
    """
    rets = log_returns(prices)

    def _tstat(s: pd.Series) -> pd.Series:
        r = s.rolling(lookback, min_periods=lookback)
        mu = r.mean()
        sigma = r.std()
        n = lookback
        return mu / (sigma / np.sqrt(n))

    return rets.groupby(level="ticker").transform(_tstat)
