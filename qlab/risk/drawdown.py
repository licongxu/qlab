"""Drawdown analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the drawdown time-series from daily returns.

    Returns
    -------
    Series
        Drawdown at each date (non-positive values; 0 at peaks).
    """
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    return cum / running_max - 1


def drawdown_details(returns: pd.Series) -> pd.DataFrame:
    """Identify individual drawdown episodes.

    Returns
    -------
    DataFrame
        Columns: ``start``, ``trough``, ``end``, ``depth``, ``days``,
        ``recovery_days``.  Rows are sorted by depth (worst first).
    """
    dd = drawdown_series(returns)
    is_dd = dd < 0

    # Identify drawdown episodes as contiguous blocks
    episodes: list[dict] = []
    in_dd = False
    start = None
    trough_date = None
    trough_val = 0.0

    for date, val in dd.items():
        if val < 0 and not in_dd:
            in_dd = True
            start = date
            trough_date = date
            trough_val = val
        elif val < 0 and in_dd:
            if val < trough_val:
                trough_date = date
                trough_val = val
        elif val >= 0 and in_dd:
            in_dd = False
            episodes.append(
                {
                    "start": start,
                    "trough": trough_date,
                    "end": date,
                    "depth": trough_val,
                }
            )

    # Handle ongoing drawdown at end of series
    if in_dd:
        episodes.append(
            {
                "start": start,
                "trough": trough_date,
                "end": dd.index[-1],
                "depth": trough_val,
            }
        )

    if not episodes:
        return pd.DataFrame(
            columns=["start", "trough", "end", "depth", "days", "recovery_days"]
        )

    df = pd.DataFrame(episodes)

    # Compute durations
    all_dates = dd.index
    date_to_pos = {d: i for i, d in enumerate(all_dates)}
    df["days"] = df.apply(
        lambda r: date_to_pos.get(r["end"], 0) - date_to_pos.get(r["start"], 0),
        axis=1,
    )
    df["recovery_days"] = df.apply(
        lambda r: date_to_pos.get(r["end"], 0) - date_to_pos.get(r["trough"], 0),
        axis=1,
    )
    return df.sort_values("depth").reset_index(drop=True)
