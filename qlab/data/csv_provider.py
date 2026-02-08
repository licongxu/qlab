"""CSV-based market data provider.

Reads per-ticker CSV files from a local directory.  Each file must contain
at least the columns ``Date, Open, High, Low, Close, Volume`` (case-insensitive).
An optional ``Adj Close`` column is used when present; otherwise ``Close`` is
copied to ``adj_close``.

This provider also exposes :func:`generate_synthetic`, which creates
realistic-looking random price data useful for unit tests and demos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from qlab.data.base import MarketDataProvider
from qlab.utils.validation import QlabValidationError


class CsvProvider(MarketDataProvider):
    """Read OHLCV data from one CSV file per ticker.

    Parameters
    ----------
    directory : str or Path
        Folder containing ``<TICKER>.csv`` files.
    date_column : str
        Name of the date column in the CSV files (default ``'Date'``).
    """

    def __init__(self, directory: str | Path, date_column: str = "Date") -> None:
        self.directory = Path(directory)
        self.date_column = date_column
        if not self.directory.is_dir():
            raise QlabValidationError(
                f"CSV directory does not exist: {self.directory}"
            )

    def fetch(
        self,
        tickers: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        parts: list[pd.DataFrame] = []
        for ticker in tickers:
            path = self.directory / f"{ticker}.csv"
            if not path.exists():
                raise QlabValidationError(f"CSV file not found: {path}")
            raw = pd.read_csv(path, parse_dates=[self.date_column])
            raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")
            raw = raw.rename(columns={"date": "date_col"})
            if "date_col" not in raw.columns:
                raw = raw.rename(
                    columns={self.date_column.lower().replace(" ", "_"): "date_col"}
                )
            raw = raw[(raw["date_col"] >= start_ts) & (raw["date_col"] <= end_ts)]
            if "adj_close" not in raw.columns:
                raw["adj_close"] = raw["close"]
            df = raw[["date_col", "open", "high", "low", "close", "volume", "adj_close"]].copy()
            df = df.rename(columns={"date_col": "date"})
            df["ticker"] = ticker
            df = df.set_index(["date", "ticker"])
            parts.append(df)
        if not parts:
            raise QlabValidationError("No data fetched — empty ticker list.")
        return pd.concat(parts).sort_index()


def generate_synthetic(
    tickers: Sequence[str],
    start: str = "2015-01-02",
    end: str = "2023-12-29",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data for testing.

    Prices follow geometric Brownian motion with per-ticker drift and
    volatility drawn from realistic distributions.

    Parameters
    ----------
    tickers : sequence of str
        Ticker symbols.
    start, end : str
        Date range boundaries.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        Stacked OHLCV with MultiIndex ``(date, ticker)``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end, freq="B")
    n_days = len(dates)
    parts: list[pd.DataFrame] = []
    for ticker in tickers:
        annual_drift = rng.uniform(0.02, 0.12)
        annual_vol = rng.uniform(0.15, 0.45)
        daily_drift = annual_drift / 252
        daily_vol = annual_vol / np.sqrt(252)
        log_returns = rng.normal(daily_drift, daily_vol, size=n_days)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        # Synthetic intraday range
        spread = rng.uniform(0.005, 0.02, size=n_days) * close
        high = close + spread * rng.uniform(0.3, 1.0, size=n_days)
        low = close - spread * rng.uniform(0.3, 1.0, size=n_days)
        low = np.maximum(low, 0.01)
        open_ = close * np.exp(rng.normal(0, daily_vol * 0.3, size=n_days))
        volume = rng.integers(100_000, 10_000_000, size=n_days).astype(float)
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "adj_close": close,
            },
            index=pd.MultiIndex.from_arrays(
                [dates, np.full(n_days, ticker)],
                names=["date", "ticker"],
            ),
        )
        parts.append(df)
    return pd.concat(parts).sort_index()
