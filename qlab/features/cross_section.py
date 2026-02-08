"""Cross-sectional transforms applied across tickers on each date.

All functions take and return a Series with MultiIndex ``(date, ticker)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank(signal: pd.Series, pct: bool = True) -> pd.Series:
    """Cross-sectional percentile rank on each date.

    Parameters
    ----------
    signal : Series
        Alpha signal.
    pct : bool
        If True (default), return ranks in ``[0, 1]``.  Otherwise integer ranks.
    """
    return signal.groupby(level="date").rank(pct=pct)


def zscore(signal: pd.Series) -> pd.Series:
    """Cross-sectional z-score: ``(x - mean) / std`` on each date."""
    def _zscore(group: pd.Series) -> pd.Series:
        mu = group.mean()
        sigma = group.std()
        if sigma == 0 or np.isnan(sigma):
            return group * 0.0
        return (group - mu) / sigma

    return signal.groupby(level="date").transform(_zscore)


def demean(signal: pd.Series) -> pd.Series:
    """Cross-sectional demeaning: ``x - mean`` on each date."""
    return signal.groupby(level="date").transform(lambda g: g - g.mean())


def winsorize(
    signal: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Cross-sectional winsorization: clip to quantile bounds on each date.

    Parameters
    ----------
    signal : Series
        Raw signal.
    lower, upper : float
        Quantile bounds in ``[0, 1]``.
    """
    def _clip(group: pd.Series) -> pd.Series:
        lo = group.quantile(lower)
        hi = group.quantile(upper)
        return group.clip(lo, hi)

    return signal.groupby(level="date").transform(_clip)


def neutralize(
    signal: pd.Series,
    groups: pd.Series,
) -> pd.Series:
    """Group-neutralize a signal (e.g. sector-neutral).

    Subtracts the group mean on each date so the signal has zero mean
    within every group.

    Parameters
    ----------
    signal : Series
        Alpha signal with MultiIndex ``(date, ticker)``.
    groups : Series
        Group labels (e.g. sector codes) with the same index.

    Returns
    -------
    Series
        Neutralized signal.
    """
    combined = pd.DataFrame({"signal": signal, "group": groups})
    group_mean = combined.groupby(
        [combined.index.get_level_values("date"), "group"]
    )["signal"].transform("mean")
    return signal - group_mean
