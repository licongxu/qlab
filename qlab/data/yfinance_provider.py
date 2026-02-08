"""Yahoo Finance data provider via the ``yfinance`` package.

Install the optional dependency with::

    pip install "qlab[yfinance]"

yfinance is well suited for US equities, ETFs, and indices.  Data is
daily-bar OHLCV with an adjusted close that accounts for splits and
dividends.

.. note::
   Yahoo Finance is a free, unofficial API.  Rate limits and data
   availability may change without notice.  For reproducible research,
   wrap this provider with :class:`~qlab.data.cache.ParquetCache` so
   that fetched data is persisted locally.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from qlab.data.base import MarketDataProvider
from qlab.utils.validation import QlabValidationError


def _require_yfinance():
    try:
        import yfinance  # noqa: F401
        return yfinance
    except ImportError:
        raise ImportError(
            "yfinance is required for YFinanceProvider.  "
            "Install it with:  pip install 'qlab[yfinance]'  "
            "or:  pip install yfinance"
        )


class YFinanceProvider(MarketDataProvider):
    """Fetch daily OHLCV data from Yahoo Finance.

    Parameters
    ----------
    auto_adjust : bool
        If True (default), use yfinance's auto-adjusted prices where
        OHLC are already split- and dividend-adjusted.  ``adj_close``
        will equal ``close`` in this mode.  If False, raw prices are
        returned and ``adj_close`` comes from Yahoo's "Adj Close" column.
    progress : bool
        Show a yfinance download progress bar (default False).

    Examples
    --------
    >>> from qlab.data import YFinanceProvider, ParquetCache
    >>> provider = ParquetCache(YFinanceProvider(), cache_dir=".market_data")
    >>> prices = provider.fetch(["AAPL", "MSFT"], "2020-01-01", "2023-12-31")
    """

    def __init__(
        self,
        auto_adjust: bool = True,
        progress: bool = False,
    ) -> None:
        self.auto_adjust = auto_adjust
        self.progress = progress

    def fetch(
        self,
        tickers: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        yf = _require_yfinance()

        tickers = list(tickers)
        if not tickers:
            raise QlabValidationError("tickers must be a non-empty sequence.")

        raw = yf.download(
            tickers=tickers,
            start=str(start),
            end=str(end),
            auto_adjust=self.auto_adjust,
            progress=self.progress,
            group_by="ticker",
            threads=True,
        )

        if raw.empty:
            raise QlabValidationError(
                f"yfinance returned no data for tickers={tickers}, "
                f"start={start}, end={end}."
            )

        parts: list[pd.DataFrame] = []

        if len(tickers) == 1:
            # yfinance returns flat columns for a single ticker
            df = self._normalize_single(raw, tickers[0])
            parts.append(df)
        else:
            # Multi-ticker: columns are a MultiIndex (ticker, field)
            for ticker in tickers:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                sub = raw[ticker].copy()
                df = self._normalize_single(sub, ticker)
                parts.append(df)

        if not parts:
            raise QlabValidationError("No valid data returned from yfinance.")

        result = pd.concat(parts).sort_index()
        return result

    def _normalize_single(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """Normalize a single-ticker DataFrame to the qlab schema."""
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        col_map = {}
        for col in df.columns:
            if col in ("open", "high", "low", "close", "volume"):
                col_map[col] = col
            elif col in ("adj_close", "adj close"):
                col_map[col] = "adj_close"

        df = df.rename(columns=col_map)

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise QlabValidationError(
                f"yfinance data for {ticker} missing columns: {sorted(missing)}"
            )

        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        df = df[["open", "high", "low", "close", "volume", "adj_close"]]
        df = df.dropna(subset=["close"])

        # Build MultiIndex
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "date"
        df["ticker"] = ticker
        df = df.reset_index().set_index(["date", "ticker"])

        return df
