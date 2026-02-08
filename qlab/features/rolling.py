"""Rolling time-series statistics computed per ticker."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ts_rolling(series: pd.Series, window: int, func: str, **kwargs) -> pd.Series:
    """Apply a rolling aggregation per ticker."""
    return series.groupby(level="ticker").transform(
        lambda s: getattr(s.rolling(window, min_periods=window), func)(**kwargs)
    )


def rolling_mean(series: pd.Series, window: int = 21) -> pd.Series:
    """Rolling arithmetic mean per ticker."""
    return _ts_rolling(series, window, "mean")


def rolling_std(series: pd.Series, window: int = 21) -> pd.Series:
    """Rolling standard deviation per ticker."""
    return _ts_rolling(series, window, "std")


def rolling_skew(series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling skewness per ticker."""
    return series.groupby(level="ticker").transform(
        lambda s: s.rolling(window, min_periods=window).skew()
    )


def rolling_kurt(series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling excess kurtosis per ticker."""
    return series.groupby(level="ticker").transform(
        lambda s: s.rolling(window, min_periods=window).kurt()
    )


def rolling_correlation(
    x: pd.Series,
    y: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling Pearson correlation between two per-ticker series.

    Both *x* and *y* must share the same MultiIndex ``(date, ticker)``.
    """
    def _corr_group(group_x: pd.Series, group_y: pd.Series) -> pd.Series:
        return group_x.rolling(window, min_periods=window).corr(group_y)

    result_parts = []
    for ticker, gx in x.groupby(level="ticker"):
        gy = y.loc[gx.index]
        gx_vals = gx.droplevel("ticker")
        gy_vals = gy.droplevel("ticker")
        corr = gx_vals.rolling(window, min_periods=window).corr(gy_vals)
        idx = pd.MultiIndex.from_arrays(
            [corr.index, np.full(len(corr), ticker)],
            names=["date", "ticker"],
        )
        result_parts.append(pd.Series(corr.values, index=idx))
    return pd.concat(result_parts).sort_index()


def rolling_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling OLS beta of each ticker's returns against a benchmark.

    Parameters
    ----------
    returns : Series
        Per-ticker returns with MultiIndex ``(date, ticker)``.
    benchmark : Series
        Single-asset benchmark return indexed by date.
    window : int
        Rolling window length.
    """
    result_parts = []
    for ticker, group in returns.groupby(level="ticker"):
        r = group.droplevel("ticker")
        b = benchmark.reindex(r.index)
        cov = r.rolling(window, min_periods=window).cov(b)
        var = b.rolling(window, min_periods=window).var()
        beta = cov / var
        idx = pd.MultiIndex.from_arrays(
            [beta.index, np.full(len(beta), ticker)],
            names=["date", "ticker"],
        )
        result_parts.append(pd.Series(beta.values, index=idx))
    return pd.concat(result_parts).sort_index()


def ewm_mean(series: pd.Series, halflife: int = 21) -> pd.Series:
    """Exponentially weighted mean per ticker."""
    return series.groupby(level="ticker").transform(
        lambda s: s.ewm(halflife=halflife, min_periods=halflife).mean()
    )
