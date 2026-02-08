"""Backtest API routes."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..models import BacktestRequest, BacktestResult, RunMeta
from ..services.backtest_runner import runner

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.post("/run", response_model=RunMeta)
async def run_backtest_endpoint(req: BacktestRequest) -> RunMeta:
    run_id = await runner.submit(req)
    return runner.get_run(run_id)


@router.get("/runs", response_model=list[RunMeta])
async def list_runs() -> list[RunMeta]:
    return runner.list_runs()


@router.get("/runs/{run_id}", response_model=RunMeta)
async def get_run(run_id: str) -> RunMeta:
    try:
        return runner.get_run(run_id)
    except KeyError:
        raise HTTPException(404, f"Run {run_id} not found")


@router.get("/runs/{run_id}/results", response_model=BacktestResult)
async def get_results(run_id: str) -> BacktestResult:
    try:
        meta = runner.get_run(run_id)
    except KeyError:
        raise HTTPException(404, f"Run {run_id} not found")
    if meta.status.value != "completed":
        raise HTTPException(400, f"Run {run_id} status is {meta.status.value}")
    try:
        return runner.get_result(run_id)
    except KeyError:
        raise HTTPException(404, f"Results for {run_id} not available")


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> dict:
    try:
        runner.get_run(run_id)
    except KeyError:
        raise HTTPException(404, f"Run {run_id} not found")
    runner._runs.pop(run_id, None)
    runner._results.pop(run_id, None)
    return {"deleted": run_id}


@router.websocket("/ws/{run_id}")
async def backtest_progress(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    try:
        meta = runner.get_run(run_id)
    except KeyError:
        await websocket.close(code=4004, reason="Run not found")
        return

    if meta.status in (RunMeta.model_fields["status"].default.__class__("completed"),):
        await websocket.send_json({"progress": 1.0, "status": "completed"})
        await websocket.close()
        return

    queue = runner.subscribe_progress(run_id)
    try:
        while True:
            try:
                progress = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
                continue

            meta = runner.get_run(run_id)
            await websocket.send_json({
                "progress": progress,
                "status": meta.status.value,
            })
            if meta.status.value in ("completed", "failed"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        runner._progress_callbacks.get(run_id, []).remove(queue) if queue in runner._progress_callbacks.get(run_id, []) else None
