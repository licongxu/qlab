import type {
  AlphaInfo,
  AlphaAnalysis,
  BacktestRequest,
  BacktestResult,
  RunMeta,
  UniverseInfo,
} from "../types/api.ts";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export const getHealth = () => request<{ status: string; version: string }>("/health");

export const submitBacktest = (req: BacktestRequest) =>
  request<RunMeta>("/backtest/run", { method: "POST", body: JSON.stringify(req) });

export const listRuns = () => request<RunMeta[]>("/backtest/runs");

export const getRun = (id: string) => request<RunMeta>(`/backtest/runs/${id}`);

export const getResults = (id: string) => request<BacktestResult>(`/backtest/runs/${id}/results`);

export const deleteRun = (id: string) =>
  request<{ deleted: string }>(`/backtest/runs/${id}`, { method: "DELETE" });

export const listAlphas = () => request<AlphaInfo[]>("/alphas/");

export const getAlpha = (name: string) => request<AlphaInfo>(`/alphas/${name}`);

export const analyzeAlpha = (name: string, params?: Record<string, unknown>) =>
  request<AlphaAnalysis>(`/alphas/${name}/analyze`, {
    method: "POST",
    body: JSON.stringify(params ?? {}),
  });

export const listUniverses = () => request<UniverseInfo[]>("/universes/");

export const getUniverse = (name: string) => request<UniverseInfo>(`/universes/${name}`);

export function connectProgress(runId: string, onMessage: (data: { progress: number; status: string }) => void) {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${window.location.host}/api/backtest/ws/${runId}`);
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type !== "heartbeat") onMessage(data);
  };
  return ws;
}
