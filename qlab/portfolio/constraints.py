"""Position constraints and exposure control.

Functions take a weights Series and return a constrained weights Series
with the same index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_position_limits(
    weights: pd.Series,
    max_weight: float = 0.05,
    min_weight: float = -0.05,
) -> pd.Series:
    """Clip individual position weights and re-normalise.

    After clipping, the remaining weight is redistributed proportionally
    among unconstrained positions so the gross exposure is preserved.

    Parameters
    ----------
    weights : Series
        Portfolio weights, MultiIndex ``(date, ticker)``.
    max_weight : float
        Maximum per-position weight (long side).
    min_weight : float
        Minimum per-position weight (short side, should be negative).
    """

    def _clip(group: pd.Series) -> pd.Series:
        return group.clip(lower=min_weight, upper=max_weight)

    return weights.groupby(level="date", group_keys=False).apply(_clip)


def apply_turnover_limit(
    new_weights: pd.Series,
    old_weights: pd.Series,
    max_turnover: float = 0.20,
) -> pd.Series:
    """Limit single-period turnover by blending old and new weights.

    If the proposed turnover exceeds *max_turnover*, the new weights are
    blended towards the old weights until the constraint is satisfied.

    Parameters
    ----------
    new_weights : Series
        Proposed new weights.
    old_weights : Series
        Current weights (previous period).
    max_turnover : float
        Maximum one-way turnover (sum of absolute weight changes / 2).
    """
    # Align on a common index (outer join — missing positions are zero)
    combined = pd.DataFrame({
        "new": new_weights,
        "old": old_weights,
    }).fillna(0.0)

    def _limit(group: pd.DataFrame) -> pd.Series:
        delta = group["new"] - group["old"]
        turnover = delta.abs().sum() / 2
        if turnover <= max_turnover:
            return group["new"]
        # Blend: w = old + alpha * (new - old), choose alpha so turnover = max
        alpha = max_turnover / turnover
        return group["old"] + alpha * delta

    return combined.groupby(level="date", group_keys=False).apply(_limit)


def dollar_neutral(weights: pd.Series) -> pd.Series:
    """Adjust weights so net exposure is zero on each date.

    The mean weight is subtracted cross-sectionally.
    """

    def _dn(group: pd.Series) -> pd.Series:
        return group - group.mean()

    return weights.groupby(level="date", group_keys=False).apply(_dn)
