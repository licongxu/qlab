"""Volatility estimators.

All estimators return an annualised volatility Series with the same
MultiIndex ``(date, ticker)`` as the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ANN_FACTOR = np.sqrt(252)


def realized_volatility(
    returns: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Rolling realized (close-to-close) volatility.

    Parameters
    ----------
    returns : Series
        Daily log or simple returns.
    window : int
        Rolling window in trading days.
    annualize : bool
        If True, multiply by sqrt(252).
    """
    vol = returns.groupby(level="ticker").transform(
        lambda s: s.rolling(window, min_periods=window).std()
    )
    if annualize:
        vol = vol * _ANN_FACTOR
    return vol


def ewm_volatility(
    returns: pd.Series,
    halflife: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Exponentially weighted moving volatility.

    Parameters
    ----------
    returns : Series
        Daily returns.
    halflife : int
        EWM half-life in trading days.
    annualize : bool
        If True, multiply by sqrt(252).
    """
    vol = returns.groupby(level="ticker").transform(
        lambda s: s.ewm(halflife=halflife, min_periods=halflife).std()
    )
    if annualize:
        vol = vol * _ANN_FACTOR
    return vol


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Parkinson (1980) high-low range volatility estimator.

    More efficient than close-to-close when intraday range is available.
    """
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2.0))
    sq = factor * log_hl ** 2
    var = sq.groupby(level="ticker").transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    vol = np.sqrt(var)
    if annualize:
        vol = vol * _ANN_FACTOR
    return vol


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Garman-Klass (1980) OHLC volatility estimator.

    Uses all four price fields for a more efficient variance estimate.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    component = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    var = component.groupby(level="ticker").transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    vol = np.sqrt(var.clip(lower=0))
    if annualize:
        vol = vol * _ANN_FACTOR
    return vol
