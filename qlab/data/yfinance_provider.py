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

        tickers = [t.strip().upper() for t in tickers if t.strip()]
        if not tickers:
            raise QlabValidationError("tickers must be a non-empty sequence.")

        try:
            raw = yf.download(
                tickers=tickers,
                start=str(start),
                end=str(end),
                auto_adjust=self.auto_adjust,
                progress=self.progress,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            raise QlabValidationError(
                f"Data source/network error fetching {tickers}: {e}"
            ) from e

        if raw.empty:
            raise QlabValidationError(
                f"No data returned for tickers={tickers}, "
                f"start={start}, end={end}. "
                "Verify ticker symbols exist on Yahoo Finance."
            )

        parts: list[pd.DataFrame] = []
        missing_tickers: list[str] = []

        # yfinance >=1.0 may return MultiIndex columns (field, ticker) even
        # for a single ticker.  Detect and handle both formats.
        has_multi_cols = isinstance(raw.columns, pd.MultiIndex)

        if len(tickers) == 1 and not has_multi_cols:
            # Legacy format: flat columns for a single ticker
            df = self._normalize_single(raw, tickers[0])
            if df.empty:
                missing_tickers.append(tickers[0])
            else:
                parts.append(df)
        else:
            # MultiIndex columns: (field, ticker) or (ticker, field)
            # Determine which level holds ticker symbols
            if has_multi_cols:
                lvl0_vals = set(raw.columns.get_level_values(0).str.upper())
                ticker_set = set(t.upper() for t in tickers)
                if lvl0_vals & ticker_set:
                    ticker_level = 0
                else:
                    ticker_level = 1
            else:
                ticker_level = 0

            for ticker in tickers:
                try:
                    if has_multi_cols:
                        if ticker_level == 0:
                            sub = raw[ticker].copy()
                        else:
                            sub = raw.xs(ticker, level=ticker_level, axis=1).copy()
                    else:
                        sub = raw[[ticker]].copy()
                except KeyError:
                    missing_tickers.append(ticker)
                    continue

                # Flatten MultiIndex columns if still multi-level
                if isinstance(sub.columns, pd.MultiIndex):
                    sub.columns = sub.columns.get_level_values(0)

                df = self._normalize_single(sub, ticker)
                if df.empty:
                    missing_tickers.append(ticker)
                else:
                    parts.append(df)

        if missing_tickers:
            import warnings

            warnings.warn(
                f"Tickers not found in data source: {missing_tickers}"
            )

        if not parts:
            raise QlabValidationError(
                f"No valid data returned from yfinance. "
                f"Tickers not found: {missing_tickers}"
            )

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
