"""Stock selection API routes."""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import SelectionConfig
from app.stock_picker import StockPicker

router = APIRouter(prefix="/api/selection", tags=["selection"])
_executor = ThreadPoolExecutor(max_workers=1)
_runs: dict[str, dict] = {}


class SelectionRequest(BaseModel):
    n_long: int = Field(20, ge=5, le=50)
    long_only: bool = True
    n_short: int = Field(0, ge=0, le=50)
    start_date: str = "2022-01-01"
    end_date: str = "2026-02-08"
    rebalance_freq: str = "weekly"
    max_position: float = Field(0.08, gt=0, le=0.5)
    sector_cap: float = Field(0.30, gt=0, le=1.0)
    commission_bps: float = Field(3.0, ge=0)
    slippage_bps: float = Field(3.0, ge=0)


@router.post("/run")
async def run_selection(req: SelectionRequest) -> dict:
    config = SelectionConfig(
        start_date=req.start_date,
        end_date=req.end_date,
        n_long=req.n_long,
        long_only=req.long_only,
        n_short=req.n_short,
        max_position_weight=req.max_position,
        sector_cap=req.sector_cap,
        rebalance_freq=req.rebalance_freq,
        commission_bps=req.commission_bps,
        slippage_bps=req.slippage_bps,
    )
    picker = StockPicker(config)
    _runs[picker.run_id] = {"status": "running", "run_id": picker.run_id}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, picker.run)
    _runs[picker.run_id] = {"status": "completed", **result}
    return _runs[picker.run_id]


@router.get("/runs")
async def list_selection_runs() -> list[dict]:
    run_dir = Path("runs")
    if not run_dir.exists():
        return []
    runs = []
    for d in sorted(run_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / "selection.csv").exists():
            info = {"run_id": d.name, "status": "completed"}
            cfg_path = d / "config.json"
            if cfg_path.exists():
                info["config"] = json.loads(cfg_path.read_text())
            bt_path = d / "backtest_summary.json"
            if bt_path.exists():
                info["backtest"] = json.loads(bt_path.read_text())
            runs.append(info)
    return runs


@router.get("/runs/{run_id}/status")
async def get_selection_status(run_id: str) -> dict:
    if run_id in _runs:
        return _runs[run_id]
    run_path = Path("runs") / run_id
    if run_path.exists():
        return {"run_id": run_id, "status": "completed"}
    raise HTTPException(404, f"Run {run_id} not found")


@router.get("/runs/{run_id}/selection")
async def get_selection(run_id: str) -> list[dict]:
    csv_path = Path("runs") / run_id / "selection.csv"
    if not csv_path.exists():
        raise HTTPException(404, f"Selection for run {run_id} not found")
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


@router.get("/runs/{run_id}/report")
async def get_report(run_id: str) -> dict:
    run_path = Path("runs") / run_id
    if not run_path.exists():
        raise HTTPException(404, f"Run {run_id} not found")

    report: dict = {"run_id": run_id}
    for fname in ["backtest_summary.json", "config.json", "regime.json", "filter_log.json"]:
        fpath = run_path / fname
        if fpath.exists():
            report[fname.replace(".json", "")] = json.loads(fpath.read_text())

    csv_path = run_path / "selection.csv"
    if csv_path.exists():
        import pandas as pd
        report["selection"] = pd.read_csv(csv_path).to_dict(orient="records")

    return report
