"""Low-volatility and defensive alpha signals.

Convention: higher signal → more attractive.
Since low-volatility is a *defensive* factor, signals are *negated*
volatility or beta so that lower-risk stocks receive a higher score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.features.returns import log_returns
from qlab.features.volatility import realized_volatility
from qlab.features.rolling import rolling_beta as _rolling_beta


def low_volatility(
    prices: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """Negative realised volatility signal.

    Lower volatility → higher signal value.

    Parameters
    ----------
    prices : Series
        Adjusted close with MultiIndex ``(date, ticker)``.
    lookback : int
        Rolling window for volatility estimation.
    """
    rets = log_returns(prices)
    vol = realized_volatility(rets, window=lookback, annualize=True)
    return -vol


def idiosyncratic_vol(
    prices: pd.Series,
    market_prices: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """Negative idiosyncratic volatility signal.

    Computes residual volatility after removing market beta exposure.

    Parameters
    ----------
    prices : Series
        Per-ticker adjusted close, MultiIndex ``(date, ticker)``.
    market_prices : Series
        Market index close, indexed by date only.
    lookback : int
        Rolling window.
    """
    rets = log_returns(prices)
    mkt_rets = market_prices.pct_change()
    beta = _rolling_beta(rets, mkt_rets, window=lookback)
    dates = rets.index.get_level_values("date")
    mkt_aligned = mkt_rets.reindex(dates).values
    residual = rets - beta * mkt_aligned
    vol = realized_volatility(residual, window=lookback, annualize=True)
    return -vol


def beta_signal(
    prices: pd.Series,
    market_prices: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """Negative market beta signal (betting against beta).

    Lower-beta stocks receive a higher signal score.

    Parameters
    ----------
    prices : Series
        Per-ticker adjusted close.
    market_prices : Series
        Market index close, indexed by date only.
    lookback : int
        Rolling window for beta estimation.
    """
    rets = log_returns(prices)
    mkt_rets = market_prices.pct_change()
    beta = _rolling_beta(rets, mkt_rets, window=lookback)
    return -beta
