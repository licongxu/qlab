"""Universe management routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models import UniverseInfo

router = APIRouter(prefix="/api/universes", tags=["universes"])

BUILTIN_UNIVERSES: dict[str, UniverseInfo] = {
    "us_large_cap_20": UniverseInfo(
        name="us_large_cap_20",
        tickers=[
            "AAPL", "MSFT", "GOOG", "AMZN", "META",
            "JPM", "GS", "BAC",
            "JNJ", "PFE", "UNH",
            "XOM", "CVX",
            "PG", "KO", "WMT",
            "HD", "NKE",
            "CAT", "HON",
        ],
        description="20 large-cap US equities across sectors",
    ),
    "mega_tech": UniverseInfo(
        name="mega_tech",
        tickers=["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA"],
        description="Mega-cap technology stocks",
    ),
    "financials": UniverseInfo(
        name="financials",
        tickers=["JPM", "GS", "BAC", "C", "WFC", "MS", "BLK", "SCHW"],
        description="Major US financial institutions",
    ),
    "healthcare": UniverseInfo(
        name="healthcare",
        tickers=["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "LLY"],
        description="Large-cap US healthcare and pharma",
    ),
    "sp500_sectors": UniverseInfo(
        name="sp500_sectors",
        tickers=[
            "XLK", "XLF", "XLV", "XLE", "XLI",
            "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC",
        ],
        description="S&P 500 sector ETFs",
    ),
}


@router.get("/", response_model=list[UniverseInfo])
async def list_universes() -> list[UniverseInfo]:
    return list(BUILTIN_UNIVERSES.values())


@router.get("/{name}", response_model=UniverseInfo)
async def get_universe(name: str) -> UniverseInfo:
    if name not in BUILTIN_UNIVERSES:
        raise HTTPException(404, f"Universe '{name}' not found")
    return BUILTIN_UNIVERSES[name]
