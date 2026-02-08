"""Tushare data provider for Chinese A-share equities.

Install the optional dependency with::

    pip install "qlab[tushare]"

Tushare Pro requires a free API token, obtainable at https://tushare.pro.

Ticker codes follow the Tushare convention: ``"000001.SZ"`` (Shenzhen),
``"600000.SH"`` (Shanghai).

.. note::
   Tushare enforces per-minute API rate limits that vary by user level.
   Wrap this provider with :class:`~qlab.data.cache.ParquetCache` to
   avoid repeated downloads.
"""

from __future__ import annotations

import time
from typing import Sequence

import pandas as pd

from qlab.data.base import MarketDataProvider
from qlab.utils.validation import QlabValidationError


def _require_tushare():
    try:
        import tushare  # noqa: F401
        return tushare
    except ImportError:
        raise ImportError(
            "tushare is required for TushareProvider.  "
            "Install it with:  pip install 'qlab[tushare]'  "
            "or:  pip install tushare"
        )


class TushareProvider(MarketDataProvider):
    """Fetch daily OHLCV data from Tushare Pro.

    Parameters
    ----------
    token : str
        Tushare Pro API token.  If not provided, falls back to the
        ``TUSHARE_TOKEN`` environment variable, then to a previously
        set token via ``tushare.set_token()``.
    adj : str
        Adjustment type: ``'qfq'`` (forward-adjusted, default),
        ``'hfq'`` (backward-adjusted), or ``''`` (unadjusted).
    pause : float
        Seconds to sleep between per-ticker API calls to respect rate
        limits (default 0.3).

    Examples
    --------
    >>> from qlab.data import TushareProvider, ParquetCache
    >>> provider = ParquetCache(
    ...     TushareProvider(token="your_token_here"),
    ...     cache_dir=".cn_market_data",
    ... )
    >>> prices = provider.fetch(
    ...     ["000001.SZ", "600519.SH"],
    ...     "2020-01-01", "2023-12-31",
    ... )
    """

    def __init__(
        self,
        token: str | None = None,
        adj: str = "qfq",
        pause: float = 0.3,
    ) -> None:
        ts = _require_tushare()
        if token is not None:
            ts.set_token(token)
        else:
            import os
            env_token = os.environ.get("TUSHARE_TOKEN")
            if env_token:
                ts.set_token(env_token)

        self._api = ts.pro_api()
        self.adj = adj
        self.pause = pause

    def fetch(
        self,
        tickers: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        ts = _require_tushare()

        tickers = list(tickers)
        if not tickers:
            raise QlabValidationError("tickers must be a non-empty sequence.")

        start_str = pd.Timestamp(start).strftime("%Y%m%d")
        end_str = pd.Timestamp(end).strftime("%Y%m%d")

        parts: list[pd.DataFrame] = []
        for i, ticker in enumerate(tickers):
            if i > 0 and self.pause > 0:
                time.sleep(self.pause)

            df = ts.pro_bar(
                ts_code=ticker,
                api=self._api,
                start_date=start_str,
                end_date=end_str,
                adj=self.adj,
                freq="D",
            )

            if df is None or df.empty:
                continue

            normalized = self._normalize(df, ticker)
            parts.append(normalized)

        if not parts:
            raise QlabValidationError(
                f"Tushare returned no data for tickers={tickers}, "
                f"start={start}, end={end}.  "
                "Check that the ticker codes are valid (e.g. '000001.SZ') "
                "and your Tushare token has sufficient permissions."
            )

        return pd.concat(parts).sort_index()

    @staticmethod
    def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Map Tushare column names to the qlab schema."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("date")

        result = pd.DataFrame({
            "open": df["open"].values,
            "high": df["high"].values,
            "low": df["low"].values,
            "close": df["close"].values,
            "volume": df["vol"].values,  # tushare: vol is in lots (手)
            "adj_close": df["close"].values,  # pre-adjusted when adj='qfq'
        }, index=pd.MultiIndex.from_arrays(
            [df["date"].values, [ticker] * len(df)],
            names=["date", "ticker"],
        ))

        return result
