"""Transparent parquet disk cache wrapping any MarketDataProvider."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd

from qlab.data.base import MarketDataProvider


class ParquetCache(MarketDataProvider):
    """Caching decorator for any :class:`MarketDataProvider`.

    On the first call for a given ``(tickers, start, end)`` triple the data
    is fetched from the underlying provider and persisted as a Parquet file.
    Subsequent calls with the same arguments return the cached copy.

    Parameters
    ----------
    provider : MarketDataProvider
        The upstream data source.
    cache_dir : str or Path
        Directory for cached Parquet files (created if absent).
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        cache_dir: str | Path = ".qlab_cache",
    ) -> None:
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self,
        tickers: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> str:
        raw = f"{sorted(tickers)}|{start}|{end}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def fetch(
        self,
        tickers: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        key = self._cache_key(tickers, start, end)
        path = self._cache_path(key)
        if path.exists():
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.MultiIndex):
                df = df.set_index(["date", "ticker"])
            return df
        df = self.provider.fetch(tickers, start, end)
        df.to_parquet(path)
        return df

    def clear(self) -> int:
        """Remove all cached files.  Returns the number of files deleted."""
        files = list(self.cache_dir.glob("*.parquet"))
        for f in files:
            f.unlink()
        return len(files)
