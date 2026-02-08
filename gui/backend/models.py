"""Pydantic models for the qlab GUI API."""
from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class RunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class RebalanceFreq(str, enum.Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"


# ── Request models ─────────────────────────────────────────────────────

class AlphaConfig(BaseModel):
    name: str = Field(..., description="Alpha function name from registry")
    params: dict[str, Any] = Field(default_factory=dict, description="Keyword arguments")
    weight: float = Field(1.0, ge=0, description="Weight in composite signal")


class BacktestRequest(BaseModel):
    name: str = Field("Untitled Strategy", max_length=200)
    tickers: list[str] = Field(..., min_length=1)
    start_date: date
    end_date: date
    alphas: list[AlphaConfig] = Field(..., min_length=1)
    long_pct: float = Field(0.2, gt=0, le=0.5)
    short_pct: float = Field(0.2, gt=0, le=0.5)
    rebalance_freq: RebalanceFreq = RebalanceFreq.monthly
    commission_bps: float = Field(5.0, ge=0)
    slippage_bps: float = Field(5.0, ge=0)
    max_position: float = Field(0.10, gt=0, le=1.0)


# ── Response models ────────────────────────────────────────────────────

class RunMeta(BaseModel):
    run_id: str
    name: str
    status: RunStatus
    created_at: datetime
    tickers: list[str]
    start_date: date
    end_date: date
    progress: float = Field(0.0, ge=0, le=1.0, description="0-1 progress")
    error: str | None = None


class PerformanceMetrics(BaseModel):
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate: float
    profit_factor: float


class TimeSeriesPoint(BaseModel):
    date: str
    value: float


class DrawdownEpisode(BaseModel):
    start: str
    trough: str
    end: str | None
    depth: float
    days: int
    recovery_days: int | None


class HoldingSnapshot(BaseModel):
    date: str
    ticker: str
    weight: float


class BacktestResult(BaseModel):
    run_id: str
    name: str
    metrics: PerformanceMetrics
    equity_curve: list[TimeSeriesPoint]
    drawdown_series: list[TimeSeriesPoint]
    rolling_sharpe: list[TimeSeriesPoint]
    monthly_returns: list[TimeSeriesPoint]
    drawdown_episodes: list[DrawdownEpisode]
    holdings: list[HoldingSnapshot]
    turnover: list[TimeSeriesPoint]
    gross_exposure: list[TimeSeriesPoint]
    net_exposure: list[TimeSeriesPoint]


class AlphaInfo(BaseModel):
    name: str
    description: str
    module: str
    params: dict[str, Any] = Field(default_factory=dict, description="Default params")


class AlphaAnalysis(BaseModel):
    name: str
    signal_stats: dict[str, float]
    ic_series: list[TimeSeriesPoint]
    ic_mean: float
    ic_std: float
    quantile_returns: dict[str, float]
    signal_distribution: list[TimeSeriesPoint]


class UniverseInfo(BaseModel):
    name: str
    tickers: list[str]
    description: str
