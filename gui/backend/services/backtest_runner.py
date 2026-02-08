"""Background backtest execution service."""
from __future__ import annotations

import asyncio
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from qlab.backtest import BacktestConfig, run_backtest
from qlab.data import ParquetCache, YFinanceProvider
from qlab.features import rank, zscore, winsorize
from qlab.portfolio import (
    equal_weight_long_short,
    normalize_weights,
    apply_position_limits,
)
from qlab.risk import (
    performance_summary,
    drawdown_series,
    drawdown_details,
)

from ..models import (
    AlphaConfig,
    BacktestRequest,
    BacktestResult,
    DrawdownEpisode,
    HoldingSnapshot,
    PerformanceMetrics,
    RunMeta,
    RunStatus,
    TimeSeriesPoint,
)
from .alpha_registry import registry


class BacktestRunner:
    """Manages backtest execution in background threads."""

    def __init__(self, max_workers: int = 2, cache_dir: str = ".qlab_cache") -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._runs: dict[str, RunMeta] = {}
        self._results: dict[str, BacktestResult] = {}
        self._progress_callbacks: dict[str, list[asyncio.Queue]] = {}
        self._cache_dir = cache_dir

    def list_runs(self) -> list[RunMeta]:
        return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def get_run(self, run_id: str) -> RunMeta:
        if run_id not in self._runs:
            raise KeyError(f"Run {run_id} not found")
        return self._runs[run_id]

    def get_result(self, run_id: str) -> BacktestResult:
        if run_id not in self._results:
            raise KeyError(f"Results for run {run_id} not available")
        return self._results[run_id]

    def subscribe_progress(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._progress_callbacks.setdefault(run_id, []).append(q)
        return q

    def _notify_progress(self, run_id: str, progress: float) -> None:
        meta = self._runs[run_id]
        meta.progress = progress
        for q in self._progress_callbacks.get(run_id, []):
            try:
                q.put_nowait(progress)
            except asyncio.QueueFull:
                pass

    async def submit(self, req: BacktestRequest) -> str:
        run_id = uuid.uuid4().hex[:12]
        meta = RunMeta(
            run_id=run_id,
            name=req.name,
            status=RunStatus.pending,
            created_at=datetime.utcnow(),
            tickers=req.tickers,
            start_date=req.start_date,
            end_date=req.end_date,
        )
        self._runs[run_id] = meta

        loop = asyncio.get_event_loop()
        loop.run_in_executor(self._executor, self._run_backtest, run_id, req)
        return run_id

    def _run_backtest(self, run_id: str, req: BacktestRequest) -> None:
        meta = self._runs[run_id]
        meta.status = RunStatus.running
        self._notify_progress(run_id, 0.0)

        try:
            # Step 1: Fetch data (30%)
            provider = ParquetCache(YFinanceProvider(), cache_dir=self._cache_dir)
            prices = provider.fetch(
                req.tickers,
                str(req.start_date),
                str(req.end_date),
            )
            close = prices["adj_close"]
            self._notify_progress(run_id, 0.3)

            # Step 2: Compute alpha signals (60%)
            signals = []
            for i, alpha_cfg in enumerate(req.alphas):
                entry = registry.get(alpha_cfg.name)
                params = {**entry.default_params, **alpha_cfg.params}

                # Some alphas need market_prices
                if alpha_cfg.name in ("idiosyncratic_vol", "beta_signal"):
                    market_prices = close.groupby(level="date").mean()
                    sig = entry.fn(close, market_prices, **params)
                else:
                    sig = entry.fn(close, **params)

                sig = zscore(winsorize(sig.dropna(), lower=0.05, upper=0.95))
                signals.append(sig * alpha_cfg.weight)

                frac = 0.3 + 0.3 * (i + 1) / len(req.alphas)
                self._notify_progress(run_id, frac)

            # Combine signals
            if len(signals) == 1:
                combined = signals[0]
            else:
                common_idx = signals[0].index
                for s in signals[1:]:
                    common_idx = common_idx.intersection(s.index)
                aligned = [s.reindex(common_idx) for s in signals]
                combined = sum(aligned)
                combined = zscore(combined.dropna())

            # Step 3: Portfolio construction (70%)
            weights = equal_weight_long_short(
                combined, long_pct=req.long_pct, short_pct=req.short_pct
            )
            weights = normalize_weights(weights, gross_exposure=2.0, net_exposure=0.0)
            weights = apply_position_limits(
                weights, max_weight=req.max_position, min_weight=-req.max_position
            )
            self._notify_progress(run_id, 0.7)

            # Step 4: Run backtest (90%)
            config = BacktestConfig(
                rebalance_freq=req.rebalance_freq.value,
                commission_bps=req.commission_bps,
                slippage_bps=req.slippage_bps,
                signal_lag=1,
                execution_price="open",
            )
            bt_result = run_backtest(weights, prices, config=config)
            self._notify_progress(run_id, 0.9)

            # Step 5: Compute analytics (100%)
            result = self._build_result(run_id, req.name, bt_result, weights)
            self._results[run_id] = result
            meta.status = RunStatus.completed
            self._notify_progress(run_id, 1.0)

        except Exception as e:
            meta.status = RunStatus.failed
            meta.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            self._notify_progress(run_id, -1.0)

    def _build_result(
        self, run_id: str, name: str, bt_result: Any, weights: pd.Series
    ) -> BacktestResult:
        port_ret = bt_result.portfolio_returns
        summary = performance_summary(port_ret)
        dd = drawdown_series(port_ret)
        dd_details = drawdown_details(port_ret)

        # Equity curve
        cum_ret = (1 + port_ret).cumprod()
        equity_curve = [
            TimeSeriesPoint(date=str(d.date()), value=round(float(v), 6))
            for d, v in cum_ret.items()
        ]

        # Drawdown series
        dd_ts = [
            TimeSeriesPoint(date=str(d.date()), value=round(float(v), 6))
            for d, v in dd.items()
        ]

        # Rolling 63-day Sharpe
        rolling_ret = port_ret.rolling(63).mean() * 252
        rolling_vol = port_ret.rolling(63).std() * np.sqrt(252)
        rolling_sr = (rolling_ret / rolling_vol).dropna()
        rolling_sharpe = [
            TimeSeriesPoint(date=str(d.date()), value=round(float(v), 4))
            for d, v in rolling_sr.items()
        ]

        # Monthly returns
        monthly = port_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly_returns = [
            TimeSeriesPoint(date=str(d.date()), value=round(float(v), 6))
            for d, v in monthly.items()
        ]

        # Drawdown episodes
        episodes = []
        if len(dd_details) > 0:
            for _, row in dd_details.head(10).iterrows():
                episodes.append(DrawdownEpisode(
                    start=str(row["start"].date()) if pd.notna(row.get("start")) else "",
                    trough=str(row["trough"].date()) if pd.notna(row.get("trough")) else "",
                    end=str(row.get("end", "").date()) if pd.notna(row.get("end")) else None,
                    depth=round(float(row["depth"]), 6),
                    days=int(row["days"]),
                    recovery_days=int(row["recovery_days"]) if pd.notna(row.get("recovery_days")) else None,
                ))

        # Holdings snapshots (sample 10 dates)
        holdings = []
        dates = weights.index.get_level_values("date").unique()
        sample_dates = dates[::max(1, len(dates) // 10)][:10]
        for d in sample_dates:
            w = weights.loc[d]
            for ticker, wt in w.items():
                if abs(wt) > 1e-6:
                    holdings.append(HoldingSnapshot(
                        date=str(d.date()), ticker=str(ticker), weight=round(float(wt), 6)
                    ))

        # Turnover
        w_wide = weights.unstack(level="ticker").fillna(0)
        turnover_s = w_wide.diff().abs().sum(axis=1).dropna()
        turnover = [
            TimeSeriesPoint(date=str(d.date()), value=round(float(v), 6))
            for d, v in turnover_s.items()
        ]

        # Exposure
        gross = weights.abs().groupby(level="date").sum()
        net = weights.groupby(level="date").sum()
        gross_exp = [
            TimeSeriesPoint(date=str(d.date()), value=round(float(v), 4))
            for d, v in gross.items()
        ]
        net_exp = [
            TimeSeriesPoint(date=str(d.date()), value=round(float(v), 4))
            for d, v in net.items()
        ]

        return BacktestResult(
            run_id=run_id,
            name=name,
            metrics=PerformanceMetrics(
                total_return=round(summary["total_return"], 6),
                annualized_return=round(summary["annualized_return"], 6),
                annualized_volatility=round(summary["annualized_volatility"], 6),
                sharpe_ratio=round(summary["sharpe_ratio"], 4),
                sortino_ratio=round(summary["sortino_ratio"], 4),
                max_drawdown=round(summary["max_drawdown"], 6),
                calmar_ratio=round(summary["calmar_ratio"], 4),
                hit_rate=round(summary["hit_rate"], 4),
                profit_factor=round(summary["profit_factor"], 4),
            ),
            equity_curve=equity_curve,
            drawdown_series=dd_ts,
            rolling_sharpe=rolling_sharpe,
            monthly_returns=monthly_returns,
            drawdown_episodes=episodes,
            holdings=holdings,
            turnover=turnover,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
        )


# Singleton
runner = BacktestRunner()
