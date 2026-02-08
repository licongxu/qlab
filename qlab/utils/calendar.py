"""Trading calendar utilities.

Uses the pandas *USFederalHolidayCalendar* combined with a custom business-day
offset to approximate the NYSE trading calendar.  For production use, consider
``pandas_market_calendars`` or ``exchange_calendars``; this lightweight
implementation avoids the extra dependency.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

_US_BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def trading_days(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return a DatetimeIndex of approximate US equity trading days."""
    return pd.date_range(start=start, end=end, freq=_US_BD)


def is_month_end(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True on the last trading day of each month."""
    s = pd.Series(dates, index=dates)
    last = s.groupby([s.dt.year, s.dt.month]).transform("last")
    return pd.Series(dates == last.values, index=dates)


def is_week_end(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True on the last trading day of each week."""
    s = pd.Series(dates, index=dates)
    last = s.groupby([s.dt.isocalendar().year, s.dt.isocalendar().week]).transform(
        "last"
    )
    return pd.Series(dates == last.values, index=dates)


def rebalance_dates(
    dates: pd.DatetimeIndex,
    freq: Literal["daily", "weekly", "monthly"] = "monthly",
) -> pd.DatetimeIndex:
    """Return the subset of *dates* that are rebalance days.

    Parameters
    ----------
    dates : DatetimeIndex
        Full set of trading days.
    freq : str
        One of ``'daily'``, ``'weekly'``, ``'monthly'``.

    Returns
    -------
    DatetimeIndex
        The dates on which rebalancing should occur.
    """
    if freq == "daily":
        return dates
    if freq == "weekly":
        mask = is_week_end(dates)
    elif freq == "monthly":
        mask = is_month_end(dates)
    else:
        raise ValueError(f"Unknown rebalance frequency: {freq!r}")
    return dates[mask.values]
