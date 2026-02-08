import { useEffect, useState } from "react";
import { Clock, Play, Trash2, TrendingUp } from "lucide-react";
import { listRuns, deleteRun, getResults } from "../lib/api.ts";
import { MetricCard } from "../components/MetricCard.tsx";
import { TSLineChart, TSAreaChart } from "../components/Chart.tsx";
import { ProgressBar } from "../components/ProgressBar.tsx";
import type { RunMeta, BacktestResult } from "../types/api.ts";

interface Props {
  onNavigateBuilder: () => void;
}

export function Dashboard({ onNavigateBuilder }: Props) {
  const [runs, setRuns] = useState<RunMeta[]>([]);
  const [selected, setSelected] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    try { setRuns(await listRuns()); } catch { /* backend offline */ }
  };

  useEffect(() => { refresh(); const id = setInterval(refresh, 3000); return () => clearInterval(id); }, []);

  const handleSelect = async (run: RunMeta) => {
    if (run.status !== "completed") return;
    setLoading(true);
    try { setSelected(await getResults(run.run_id)); } catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  const handleDelete = async (e: React.MouseEvent, runId: string) => {
    e.stopPropagation();
    await deleteRun(runId);
    if (selected?.run_id === runId) setSelected(null);
    refresh();
  };

  const m = selected?.metrics;

  return (
    <div className="flex-1 p-6 overflow-y-auto" style={{ background: "var(--bg)" }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold gradient-text">Dashboard</h1>
          <p className="text-sm mt-1" style={{ color: "var(--fg-muted)" }}>View and compare backtest results</p>
        </div>
        <button onClick={onNavigateBuilder}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold"
          style={{
            background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
            color: "#0a0e17", border: "none", cursor: "pointer",
            boxShadow: "0 0 20px rgba(0, 212, 170, 0.25)",
          }}>
          <Play size={16} /> New Strategy
        </button>
      </div>

      <div className="flex gap-6">
        {/* Run list */}
        <div className="w-80 shrink-0 space-y-2">
          <div className="text-[10px] font-semibold uppercase tracking-[0.15em] mb-3"
            style={{ color: "var(--fg-muted)" }}>
            Backtest Runs ({runs.length})
          </div>
          {runs.length === 0 && (
            <div className="text-center py-12 rounded-xl glow-card" style={{ background: "var(--card)" }}>
              <TrendingUp size={32} style={{ color: "var(--fg-muted)", margin: "0 auto 12px" }} />
              <p className="text-sm" style={{ color: "var(--fg-muted)" }}>No backtests yet</p>
              <p className="text-xs mt-1" style={{ color: "var(--fg-muted)" }}>Click "New Strategy" to start</p>
            </div>
          )}
          {runs.map((run) => (
            <div key={run.run_id} onClick={() => handleSelect(run)}
              className="p-3.5 rounded-xl cursor-pointer glow-card"
              style={{
                background: selected?.run_id === run.run_id ? "var(--bg-elevated)" : "var(--card)",
                borderLeft: selected?.run_id === run.run_id ? "2px solid var(--accent)" : "2px solid transparent",
              }}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-sm font-medium truncate" style={{ color: "var(--fg)" }}>{run.name}</span>
                <button onClick={(e) => handleDelete(e, run.run_id)}
                  style={{ background: "none", border: "none", cursor: "pointer", color: "var(--fg-muted)", padding: 4 }}>
                  <Trash2 size={13} />
                </button>
              </div>
              <div className="flex items-center gap-2 text-[11px]" style={{ color: "var(--fg-muted)" }}>
                <Clock size={11} />
                {new Date(run.created_at).toLocaleString()}
              </div>
              <div className="flex items-center gap-2 mt-2">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                  style={{
                    background: run.status === "completed" ? "rgba(0, 212, 170, 0.15)"
                      : run.status === "failed" ? "rgba(255, 71, 87, 0.15)"
                      : "rgba(255, 165, 2, 0.15)",
                    color: run.status === "completed" ? "var(--success)"
                      : run.status === "failed" ? "var(--danger)"
                      : "var(--warning)",
                    border: `1px solid ${run.status === "completed" ? "rgba(0, 212, 170, 0.3)" : run.status === "failed" ? "rgba(255, 71, 87, 0.3)" : "rgba(255, 165, 2, 0.3)"}`,
                  }}>
                  {run.status}
                </span>
                <span className="text-[11px] font-mono" style={{ color: "var(--fg-muted)" }}>{run.tickers.length} tickers</span>
              </div>
              {(run.status === "pending" || run.status === "running") && (
                <div className="mt-2.5"><ProgressBar progress={run.progress} status={run.status} /></div>
              )}
            </div>
          ))}
        </div>

        {/* Results panel */}
        <div className="flex-1 min-w-0">
          {!selected && !loading && (
            <div className="flex items-center justify-center h-72 rounded-xl grid-bg glow-card"
              style={{ background: "var(--card)" }}>
              <div className="text-center">
                <div className="w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center"
                  style={{ background: "var(--bg-tertiary)" }}>
                  <TrendingUp size={28} style={{ color: "var(--fg-muted)" }} />
                </div>
                <p className="text-sm" style={{ color: "var(--fg-secondary)" }}>Select a completed run to view results</p>
                <p className="text-xs mt-1" style={{ color: "var(--fg-muted)" }}>Or create a new strategy to get started</p>
              </div>
            </div>
          )}
          {loading && (
            <div className="text-center py-20 rounded-xl glow-card animate-pulse-glow" style={{ background: "var(--card)" }}>
              <div className="text-sm font-mono" style={{ color: "var(--accent)" }}>Loading results...</div>
            </div>
          )}
          {selected && m && (
            <div className="space-y-5">
              <div className="flex items-center gap-3">
                <h2 className="text-lg font-bold" style={{ color: "var(--fg)" }}>{selected.name}</h2>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                  style={{ background: "var(--accent-dim)", color: "var(--accent)", border: "1px solid rgba(0, 212, 170, 0.2)" }}>
                  {selected.run_id}
                </span>
              </div>

              {/* Metric cards */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <MetricCard label="Total Return" value={(m.total_return * 100).toFixed(2) + "%"} positive={m.total_return > 0} />
                <MetricCard label="Sharpe Ratio" value={m.sharpe_ratio.toFixed(2)} positive={m.sharpe_ratio > 0} />
                <MetricCard label="Max Drawdown" value={(m.max_drawdown * 100).toFixed(2) + "%"} positive={false} />
                <MetricCard label="Ann. Volatility" value={(m.annualized_volatility * 100).toFixed(1) + "%"} />
                <MetricCard label="Ann. Return" value={(m.annualized_return * 100).toFixed(2) + "%"} positive={m.annualized_return > 0} />
                <MetricCard label="Sortino Ratio" value={m.sortino_ratio.toFixed(2)} positive={m.sortino_ratio > 0} />
                <MetricCard label="Hit Rate" value={(m.hit_rate * 100).toFixed(1) + "%"} positive={m.hit_rate > 0.5} />
                <MetricCard label="Calmar Ratio" value={m.calmar_ratio.toFixed(2)} positive={m.calmar_ratio > 0} />
              </div>

              {/* Equity Curve */}
              <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
                <TSLineChart data={selected.equity_curve} title="Equity Curve" color="#00d4aa" referenceLine={1.0}
                  formatValue={(v) => v.toFixed(3)} />
              </div>

              {/* Drawdown */}
              <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
                <TSAreaChart data={selected.drawdown_series} title="Underwater Chart" color="#ff4757" />
              </div>

              {/* 2-col charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
                  <TSLineChart data={selected.rolling_sharpe} title="Rolling 63d Sharpe" color="#8b5cf6"
                    height={220} referenceLine={0} formatValue={(v) => v.toFixed(2)} />
                </div>
                <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
                  <TSLineChart data={selected.turnover} title="Daily Turnover" color="#ffa502"
                    height={220} formatValue={(v) => v.toFixed(3)} />
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
                  <TSLineChart data={selected.gross_exposure} title="Gross Exposure" color="#3b82f6"
                    height={220} formatValue={(v) => v.toFixed(2)} />
                </div>
                <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
                  <TSLineChart data={selected.net_exposure} title="Net Exposure" color="#00d4aa"
                    height={220} referenceLine={0} formatValue={(v) => v.toFixed(3)} />
                </div>
              </div>

              {/* Drawdown episodes */}
              {selected.drawdown_episodes.length > 0 && (
                <div className="rounded-xl p-5 glow-card overflow-x-auto" style={{ background: "var(--card)" }}>
                  <div className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: "var(--fg)" }}>
                    <div className="w-1 h-4 rounded-full" style={{ background: "var(--danger)" }} />
                    Worst Drawdown Episodes
                  </div>
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr style={{ borderBottom: "1px solid var(--border)" }}>
                        {["Start", "Trough", "Depth", "Days", "Recovery"].map((h) => (
                          <th key={h} className="py-2.5 px-3 text-left text-[10px] uppercase tracking-wider"
                            style={{ color: "var(--fg-muted)" }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {selected.drawdown_episodes.slice(0, 5).map((ep, i) => (
                        <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                          <td className="py-2.5 px-3" style={{ color: "var(--fg-secondary)" }}>{ep.start}</td>
                          <td className="py-2.5 px-3" style={{ color: "var(--fg-secondary)" }}>{ep.trough}</td>
                          <td className="py-2.5 px-3" style={{ color: "var(--danger)" }}>{(ep.depth * 100).toFixed(2)}%</td>
                          <td className="py-2.5 px-3" style={{ color: "var(--fg-secondary)" }}>{ep.days}</td>
                          <td className="py-2.5 px-3" style={{ color: "var(--fg-secondary)" }}>{ep.recovery_days ?? "ongoing"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
