"""Market data ingestion, providers, and caching."""

from qlab.data.base import MarketDataProvider
from qlab.data.csv_provider import CsvProvider
from qlab.data.cache import ParquetCache

__all__ = [
    "MarketDataProvider",
    "CsvProvider",
    "ParquetCache",
    "YFinanceProvider",
    "TushareProvider",
]


def __getattr__(name: str):
    """Lazy-import optional providers so missing deps don't break the package."""
    if name == "YFinanceProvider":
        from qlab.data.yfinance_provider import YFinanceProvider
        return YFinanceProvider
    if name == "TushareProvider":
        from qlab.data.tushare_provider import TushareProvider
        return TushareProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
