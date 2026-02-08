"""Abstract market data provider interface."""

from __future__ import annotations

import abc
from typing import Sequence

import pandas as pd


class MarketDataProvider(abc.ABC):
    """Base class for all market data providers.

    Subclasses must implement :meth:`fetch`, which returns a stacked
    DataFrame with MultiIndex ``(date, ticker)`` and columns
    ``open, high, low, close, volume, adj_close``.
    """

    @abc.abstractmethod
    def fetch(
        self,
        tickers: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch OHLCV + adjusted close for the given tickers and date range.

        Parameters
        ----------
        tickers : sequence of str
            Equity symbols (e.g. ``['AAPL', 'MSFT']``).
        start, end : str or Timestamp
            Inclusive date boundaries.

        Returns
        -------
        DataFrame
            MultiIndex ``(date, ticker)`` with columns
            ``open, high, low, close, volume, adj_close``.
        """
