"""Input validation helpers.

Every public function in the library calls these at entry points to produce
clear, early error messages rather than cryptic pandas/numpy exceptions downstream.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

PRICE_COLUMNS = {"open", "high", "low", "close", "volume"}


class QlabValidationError(ValueError):
    """Raised when input data violates expected invariants."""


def _check_multiindex(df_or_series: pd.DataFrame | pd.Series, name: str) -> None:
    idx = df_or_series.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels != 2:
        raise QlabValidationError(
            f"{name} must have a 2-level MultiIndex (date, ticker); "
            f"got {type(idx).__name__} with {getattr(idx, 'nlevels', 1)} level(s)."
        )
    if not pd.api.types.is_datetime64_any_dtype(idx.get_level_values(0)):
        raise QlabValidationError(
            f"{name} level-0 ('date') must be datetime64; "
            f"got {idx.get_level_values(0).dtype}."
        )


def validate_prices(prices: pd.DataFrame) -> None:
    """Validate a stacked price DataFrame.

    Expected shape: MultiIndex (date, ticker) with at least
    columns *open, high, low, close, volume*.  ``adj_close`` is optional.

    Raises :class:`QlabValidationError` on any violation.
    """
    if not isinstance(prices, pd.DataFrame):
        raise QlabValidationError(
            f"prices must be a DataFrame; got {type(prices).__name__}."
        )
    _check_multiindex(prices, "prices")
    missing = PRICE_COLUMNS - set(prices.columns)
    if missing:
        raise QlabValidationError(
            f"prices DataFrame missing required columns: {sorted(missing)}."
        )
    close = prices["close"]
    if (close <= 0).any():
        n_bad = int((close <= 0).sum())
        raise QlabValidationError(
            f"prices['close'] contains {n_bad} non-positive value(s)."
        )


def validate_signal(signal: pd.Series) -> None:
    """Validate an alpha signal Series.

    Must be a :class:`pandas.Series` with a 2-level MultiIndex (date, ticker)
    and finite numeric values (NaN is allowed for missing data).
    """
    if not isinstance(signal, pd.Series):
        raise QlabValidationError(
            f"signal must be a Series; got {type(signal).__name__}."
        )
    _check_multiindex(signal, "signal")
    if not np.issubdtype(signal.dtype, np.floating) and not np.issubdtype(
        signal.dtype, np.integer
    ):
        raise QlabValidationError(
            f"signal dtype must be numeric; got {signal.dtype}."
        )


def validate_weights(weights: pd.Series) -> None:
    """Validate a portfolio weights Series.

    Same structure as a signal, but additionally checks that per-date
    absolute weight sums are finite.
    """
    validate_signal(weights)
    abs_sum = weights.abs().groupby(level=0).sum()
    if not np.isfinite(abs_sum.values).all():
        raise QlabValidationError(
            "weights contain non-finite per-date absolute sums."
        )
