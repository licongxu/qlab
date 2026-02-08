"""Alpha signal API routes."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from scipy.stats import spearmanr

from qlab.data import ParquetCache, YFinanceProvider
from qlab.features import simple_returns, rank, zscore, winsorize

from ..models import AlphaAnalysis, AlphaInfo, TimeSeriesPoint
from ..services.alpha_registry import registry

router = APIRouter(prefix="/api/alphas", tags=["alphas"])


@router.get("/", response_model=list[AlphaInfo])
async def list_alphas() -> list[AlphaInfo]:
    return [
        AlphaInfo(
            name=e.name,
            description=e.description,
            module=e.module,
            params=e.default_params,
        )
        for e in registry.list_all()
    ]


@router.get("/{name}", response_model=AlphaInfo)
async def get_alpha(name: str) -> AlphaInfo:
    try:
        e = registry.get(name)
    except KeyError:
        raise HTTPException(404, f"Alpha '{name}' not found")
    return AlphaInfo(
        name=e.name, description=e.description,
        module=e.module, params=e.default_params,
    )


@router.post("/{name}/analyze", response_model=AlphaAnalysis)
async def analyze_alpha(
    name: str,
    tickers: list[str] | None = None,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    params: dict[str, Any] | None = None,
) -> AlphaAnalysis:
    try:
        entry = registry.get(name)
    except KeyError:
        raise HTTPException(404, f"Alpha '{name}' not found")

    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META",
                    "JPM", "GS", "BAC", "JNJ", "PFE",
                    "XOM", "CVX", "PG", "KO", "WMT"]

    alpha_params = {**entry.default_params, **(params or {})}

    provider = ParquetCache(YFinanceProvider(), cache_dir=".qlab_cache")
    prices = provider.fetch(tickers, start_date, end_date)
    close = prices["adj_close"]

    # Compute signal
    if name in ("idiosyncratic_vol", "beta_signal"):
        mkt = close.groupby(level="date").mean()
        signal = entry.fn(close, mkt, **alpha_params)
    else:
        signal = entry.fn(close, **alpha_params)

    signal = signal.dropna()
    normed = zscore(winsorize(signal, lower=0.05, upper=0.95))

    # Signal statistics
    signal_stats = {
        "mean": round(float(signal.mean()), 6),
        "std": round(float(signal.std()), 6),
        "skew": round(float(signal.skew()), 4),
        "kurtosis": round(float(signal.kurt()), 4),
        "min": round(float(signal.min()), 6),
        "max": round(float(signal.max()), 6),
        "pct_positive": round(float((signal > 0).mean()), 4),
    }

    # IC series (rank correlation with 21-day forward returns)
    fwd_ret = simple_returns(close, periods=21).dropna()
    common = normed.index.intersection(fwd_ret.index)
    sig_aligned = normed.reindex(common)
    ret_aligned = fwd_ret.reindex(common)

    dates = common.get_level_values("date").unique()
    ic_values = []
    for d in dates:
        try:
            s = sig_aligned.loc[d]
            r = ret_aligned.loc[d]
            if len(s) >= 5:
                corr, _ = spearmanr(s.values, r.values)
                if np.isfinite(corr):
                    ic_values.append(TimeSeriesPoint(date=str(d.date()), value=round(corr, 6)))
        except (KeyError, ValueError):
            continue

    ic_vals = [p.value for p in ic_values]
    ic_mean = round(float(np.mean(ic_vals)), 6) if ic_vals else 0.0
    ic_std = round(float(np.std(ic_vals)), 6) if ic_vals else 0.0

    # Quantile returns (quintile buckets)
    quantile_returns: dict[str, float] = {}
    try:
        ranked = rank(normed)
        for q, (lo, hi) in enumerate([(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)], 1):
            mask = (ranked >= lo) & (ranked < hi)
            q_ret = ret_aligned.reindex(mask[mask].index).dropna()
            quantile_returns[f"Q{q}"] = round(float(q_ret.mean()) * 252, 6)
    except Exception:
        pass

    # Signal distribution (histogram-like)
    hist_vals, bin_edges = np.histogram(normed.values, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    signal_distribution = [
        TimeSeriesPoint(date=f"{c:.3f}", value=float(v))
        for c, v in zip(bin_centers, hist_vals)
    ]

    return AlphaAnalysis(
        name=name,
        signal_stats=signal_stats,
        ic_series=ic_values,
        ic_mean=ic_mean,
        ic_std=ic_std,
        quantile_returns=quantile_returns,
        signal_distribution=signal_distribution,
    )
