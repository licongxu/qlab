"""Signal-to-weight mapping functions.

All functions accept a signal Series with MultiIndex ``(date, ticker)``
and return a weights Series with the same index.  Weights are signed:
positive = long, negative = short.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.utils.validation import validate_signal


def equal_weight_long_short(
    signal: pd.Series,
    long_pct: float = 0.2,
    short_pct: float = 0.2,
) -> pd.Series:
    """Long top quantile, short bottom quantile, equal weight within each leg.

    Parameters
    ----------
    signal : Series
        Alpha signal, MultiIndex ``(date, ticker)``.
    long_pct : float
        Fraction of the cross-section to go long (top quintile by default).
    short_pct : float
        Fraction of the cross-section to go short (bottom quintile).

    Returns
    -------
    Series
        Portfolio weights summing to zero gross notional per date.
    """
    validate_signal(signal)
    pct_rank = signal.groupby(level="date").rank(pct=True)

    long_mask = pct_rank >= (1 - long_pct)
    short_mask = pct_rank <= short_pct

    def _assign_weights(date_group: pd.Series) -> pd.Series:
        date = date_group.index.get_level_values("date")[0]
        pct_r = pct_rank.loc[date]
        longs = pct_r >= (1 - long_pct)
        shorts = pct_r <= short_pct
        n_long = longs.sum()
        n_short = shorts.sum()
        w = pd.Series(0.0, index=date_group.index)
        if n_long > 0:
            w[longs.values] = 1.0 / n_long
        if n_short > 0:
            w[shorts.values] = -1.0 / n_short
        return w

    weights = signal.groupby(level="date", group_keys=False).apply(
        lambda g: _ew_weights(g, signal, long_pct, short_pct)
    )
    return weights


def _ew_weights(
    group: pd.Series,
    signal: pd.Series,
    long_pct: float,
    short_pct: float,
) -> pd.Series:
    """Helper: compute equal weights for a single date."""
    r = group.rank(pct=True)
    w = pd.Series(0.0, index=group.index)
    long_mask = r >= (1 - long_pct)
    short_mask = r <= short_pct
    n_long = long_mask.sum()
    n_short = short_mask.sum()
    if n_long > 0:
        w[long_mask] = 1.0 / n_long
    if n_short > 0:
        w[short_mask] = -1.0 / n_short
    return w


def quantile_weights(
    signal: pd.Series,
    n_quantiles: int = 5,
    long_quantile: int = 5,
    short_quantile: int = 1,
) -> pd.Series:
    """Long one quantile, short another, equal weight within each.

    Parameters
    ----------
    signal : Series
        Alpha signal.
    n_quantiles : int
        Number of quantile buckets.
    long_quantile : int
        Quantile number to go long (1 = lowest signal).
    short_quantile : int
        Quantile number to go short.
    """
    validate_signal(signal)

    def _qw(group: pd.Series) -> pd.Series:
        q = pd.qcut(group.rank(method="first"), n_quantiles, labels=False) + 1
        w = pd.Series(0.0, index=group.index)
        long_mask = q == long_quantile
        short_mask = q == short_quantile
        n_long = long_mask.sum()
        n_short = short_mask.sum()
        if n_long > 0:
            w[long_mask] = 1.0 / n_long
        if n_short > 0:
            w[short_mask] = -1.0 / n_short
        return w

    return signal.groupby(level="date", group_keys=False).apply(_qw)


def proportional_weights(
    signal: pd.Series,
    long_only: bool = False,
) -> pd.Series:
    """Weights proportional to signal magnitude.

    Parameters
    ----------
    signal : Series
        Alpha signal.
    long_only : bool
        If True, clip negative signals to zero before normalising.
    """
    validate_signal(signal)
    s = signal.copy()
    if long_only:
        s = s.clip(lower=0)

    def _norm(group: pd.Series) -> pd.Series:
        total = group.abs().sum()
        if total == 0:
            return group * 0.0
        return group / total

    return s.groupby(level="date", group_keys=False).apply(_norm)


def normalize_weights(
    weights: pd.Series,
    gross_exposure: float = 2.0,
    net_exposure: float = 0.0,
) -> pd.Series:
    """Rescale weights to target gross and net exposure.

    Parameters
    ----------
    weights : Series
        Raw portfolio weights.
    gross_exposure : float
        Target sum of absolute weights per date (default 2.0 = 1× long + 1× short).
    net_exposure : float
        Target net weight per date (default 0.0 = dollar neutral).
    """

    def _rescale(group: pd.Series) -> pd.Series:
        w = group.copy()
        # Shift to target net exposure
        current_net = w.sum()
        n = len(w)
        if n > 0:
            w = w - (current_net - net_exposure) / n
        # Scale to target gross exposure
        current_gross = w.abs().sum()
        if current_gross > 0:
            w = w * (gross_exposure / current_gross)
        return w

    return weights.groupby(level="date", group_keys=False).apply(_rescale)
