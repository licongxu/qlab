"""Data alignment and reshaping helpers.

Convert between *wide* (date × ticker columns) and *stacked*
(MultiIndex ``[date, ticker]``) representations, and align multiple
DataFrames/Series on a common index.
"""

from __future__ import annotations

import pandas as pd

from qlab.utils.validation import QlabValidationError


def stack_prices(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide DataFrame (dates as rows, tickers as columns) to stacked form.

    The input must have a DatetimeIndex as its index.  Each column is
    interpreted as a ticker, and every cell as the *close* price.  For full
    OHLCV stacking pass a dict of wide DataFrames and call this once per
    field, then :func:`pandas.concat` the results.

    Returns a DataFrame with MultiIndex ``(date, ticker)`` and a single
    column ``'close'``.
    """
    if not isinstance(wide.index, pd.DatetimeIndex):
        raise QlabValidationError(
            "wide DataFrame must have a DatetimeIndex as its index."
        )
    stacked = wide.stack()
    stacked.index.names = ["date", "ticker"]
    return stacked.to_frame(name="close")


def unstack_to_wide(
    stacked: pd.DataFrame | pd.Series,
    column: str | None = None,
) -> pd.DataFrame:
    """Unstack a stacked Series or single-column DataFrame to wide form.

    Parameters
    ----------
    stacked : DataFrame or Series
        Must have a 2-level MultiIndex ``(date, ticker)``.
    column : str, optional
        If *stacked* is a DataFrame, which column to unstack.
        Ignored for Series input.

    Returns
    -------
    DataFrame
        Rows are dates, columns are tickers.
    """
    if isinstance(stacked, pd.DataFrame):
        if column is None:
            if stacked.shape[1] != 1:
                raise QlabValidationError(
                    "Specify 'column' when the DataFrame has more than one column."
                )
            series = stacked.iloc[:, 0]
        else:
            series = stacked[column]
    else:
        series = stacked
    return series.unstack(level="ticker")


def align_frames(
    *frames: pd.DataFrame | pd.Series,
    how: str = "inner",
) -> tuple[pd.DataFrame | pd.Series, ...]:
    """Align multiple DataFrames / Series on a common index.

    Parameters
    ----------
    *frames : DataFrame or Series
        Any number of frames sharing the same index structure.
    how : str
        Join type: ``'inner'``, ``'outer'``, ``'left'``, ``'right'``.

    Returns
    -------
    tuple
        Aligned copies of each input.
    """
    if len(frames) < 2:
        return frames
    common_idx = frames[0].index
    for f in frames[1:]:
        if how == "inner":
            common_idx = common_idx.intersection(f.index)
        elif how == "outer":
            common_idx = common_idx.union(f.index)
        elif how == "left":
            pass  # keep first frame's index
        elif how == "right":
            common_idx = f.index
        else:
            raise ValueError(f"Unknown join type: {how!r}")
    return tuple(f.reindex(common_idx) for f in frames)
