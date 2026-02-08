"""Mean-reversion alpha signals.

Convention: higher signal → more attractive (e.g. oversold → expect bounce).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.features.rolling import rolling_mean, rolling_std


def mean_reversion_zscore(
    prices: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Distance of current price from its rolling mean, in standard deviations.

    A negative value means the price is below its recent average, suggesting
    an oversold condition (long signal in a mean-reversion framework).

    The sign is *negated* so that oversold stocks have a *positive* signal.

    Parameters
    ----------
    prices : Series
        Adjusted close with MultiIndex ``(date, ticker)``.
    lookback : int
        Rolling window in days.
    """
    ma = rolling_mean(prices, window=lookback)
    sd = rolling_std(prices, window=lookback)
    z = (prices - ma) / sd.replace(0, np.nan)
    return -z  # oversold (negative z) → positive signal


def rsi_signal(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Relative Strength Index, rescaled to ``[-1, 1]``.

    RSI < 30 is traditionally oversold → positive signal; RSI > 70 is
    overbought → negative signal.  Mapping: ``signal = (50 - RSI) / 50``.

    Parameters
    ----------
    prices : Series
        Adjusted close prices.
    period : int
        RSI look-back period.
    """
    delta = prices.groupby(level="ticker").diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(level="ticker").transform(
        lambda s: s.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    )
    avg_loss = loss.groupby(level="ticker").transform(
        lambda s: s.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    )
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return (50 - rsi) / 50  # [-1, 1]


def bollinger_signal(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Bollinger Band %B inverted: distance from upper band.

    Returns a signal where prices near the lower band yield a positive
    value (buy signal) and prices near the upper band yield negative.

    Parameters
    ----------
    prices : Series
        Adjusted close.
    window : int
        Rolling window for the moving average and std.
    num_std : float
        Number of standard deviations for band width.
    """
    ma = rolling_mean(prices, window=window)
    sd = rolling_std(prices, window=window)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    bandwidth = upper - lower
    pct_b = (prices - lower) / bandwidth.replace(0, np.nan)
    return -(pct_b - 0.5)  # center at 0, negative when overbought
