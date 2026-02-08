"""Price-based quality proxy signals.

True quality factors (ROE, accruals, etc.) require fundamental data.
These proxies use only prices and volume to approximate quality traits.

Convention: higher signal → more attractive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.features.returns import log_returns
from qlab.features.volatility import realized_volatility
from qlab.features.rolling import rolling_mean


def profitability_proxy(
    prices: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """Risk-adjusted return (Sharpe-like) as a profitability proxy.

    Stocks that have generated high returns *per unit of risk* are assumed
    to be higher-quality businesses.

    Parameters
    ----------
    prices : Series
        Adjusted close with MultiIndex ``(date, ticker)``.
    lookback : int
        Rolling window in trading days.
    """
    rets = log_returns(prices)
    mu = rolling_mean(rets, window=lookback)
    vol = realized_volatility(rets, window=lookback, annualize=False)
    return mu / vol.replace(0, np.nan)


def stability(
    prices: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """Earnings stability proxy: R-squared of log-price against a linear trend.

    High R² means the stock's price path is well-explained by a simple
    trend — a hallmark of stable, predictable businesses.

    Parameters
    ----------
    prices : Series
        Adjusted close.
    lookback : int
        Rolling window.
    """
    log_p = np.log(prices)

    def _r2(s: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=s.index)
        vals = s.values
        x = np.arange(lookback, dtype=float)
        x_demean = x - x.mean()
        ss_x = (x_demean ** 2).sum()
        for i in range(lookback, len(vals) + 1):
            window = vals[i - lookback: i]
            if np.any(np.isnan(window)):
                continue
            y_demean = window - window.mean()
            ss_y = (y_demean ** 2).sum()
            if ss_y == 0:
                out.iloc[i - 1] = 1.0
                continue
            beta = (x_demean * y_demean).sum() / ss_x
            ss_reg = beta ** 2 * ss_x
            out.iloc[i - 1] = ss_reg / ss_y
        return out

    return log_p.groupby(level="ticker").apply(
        _r2, include_groups=False
    ).droplevel(0)
