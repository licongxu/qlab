"""Input validation helpers.

Every public function in the library calls these at entry points to produce
clear, early error messages rather than cryptic pandas/numpy exceptions downstream.

Data sources
------------
The default provider :class:`~qlab.data.yfinance_provider.YFinanceProvider` uses
Yahoo Finance (unofficial API, daily bars).  Prices are auto-adjusted for splits
and dividends when ``auto_adjust=True`` (default), meaning OHLC are already
adjusted and ``adj_close == close``.  Wrap with
:class:`~qlab.data.cache.ParquetCache` for reproducibility.
"""

from __future__ import annotations

import warnings

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


def validate_market_data(
    prices: pd.DataFrame,
    max_missing_rate: float = 0.05,
) -> dict:
    """Validate market data integrity for the real-data pipeline.

    Checks
    ------
    - MultiIndex (date, ticker) structure
    - Monotonic datetime index per ticker, no duplicate rows
    - Positive close prices, high >= low
    - Non-negative volume
    - Per-ticker missing-rate below *max_missing_rate*

    Returns a dict ``{"valid": bool, "global_issues": [...], "ticker_issues": {...}}``.
    Emits :mod:`warnings` for each problem found.
    """
    _check_multiindex(prices, "prices")

    global_issues: list[str] = []
    ticker_issues: dict[str, list[str]] = {}

    # Duplicate index entries
    if prices.index.duplicated().any():
        n_dup = int(prices.index.duplicated().sum())
        msg = f"Found {n_dup} duplicate index entries"
        global_issues.append(msg)
        warnings.warn(f"Data integrity: {msg}")

    tickers = prices.index.get_level_values("ticker").unique()
    for t in tickers:
        t_data = prices.xs(t, level="ticker")
        t_issues: list[str] = []

        if not t_data.index.is_monotonic_increasing:
            t_issues.append("non-monotonic date index")

        if "close" in t_data.columns:
            neg = int((t_data["close"] <= 0).sum())
            if neg > 0:
                t_issues.append(f"{neg} non-positive close prices")
            missing = float(t_data["close"].isna().mean())
            if missing > max_missing_rate:
                t_issues.append(
                    f"close missing rate {missing:.1%} exceeds {max_missing_rate:.1%}"
                )

        if {"high", "low"} <= set(t_data.columns):
            bad = int((t_data["high"] < t_data["low"]).sum())
            if bad > 0:
                t_issues.append(f"{bad} rows where high < low")

        if "volume" in t_data.columns:
            neg_vol = int((t_data["volume"] < 0).sum())
            if neg_vol > 0:
                t_issues.append(f"{neg_vol} negative volume entries")

        if t_issues:
            ticker_issues[t] = t_issues
            for issue in t_issues:
                warnings.warn(f"Data integrity [{t}]: {issue}")

    return {
        "valid": len(global_issues) == 0 and len(ticker_issues) == 0,
        "global_issues": global_issues,
        "ticker_issues": ticker_issues,
    }
