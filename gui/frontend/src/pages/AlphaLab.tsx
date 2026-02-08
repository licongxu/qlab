import { useEffect, useState } from "react";
import { FlaskConical, Loader } from "lucide-react";
import { listAlphas, analyzeAlpha } from "../lib/api.ts";
import { MetricCard } from "../components/MetricCard.tsx";
import { TSLineChart, SimpleBarChart } from "../components/Chart.tsx";
import type { AlphaInfo, AlphaAnalysis } from "../types/api.ts";

export function AlphaLab() {
  const [alphas, setAlphas] = useState<AlphaInfo[]>([]);
  const [selected, setSelected] = useState("");
  const [analysis, setAnalysis] = useState<AlphaAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    listAlphas().then((data) => {
      setAlphas(data);
      if (data.length > 0) setSelected(data[0].name);
    }).catch(() => {});
  }, []);

  const handleAnalyze = async () => {
    if (!selected) return;
    setLoading(true); setError(""); setAnalysis(null);
    try { setAnalysis(await analyzeAlpha(selected)); }
    catch (e) { setError(String(e)); }
    finally { setLoading(false); }
  };

  const info = alphas.find((a) => a.name === selected);
  const stats = analysis?.signal_stats;
  const qr = analysis?.quantile_returns;

  return (
    <div className="flex-1 p-6 overflow-y-auto" style={{ background: "var(--bg)" }}>
      <h1 className="text-2xl font-bold gradient-text mb-1">Alpha Lab</h1>
      <p className="text-sm mb-8" style={{ color: "var(--fg-muted)" }}>Analyze individual alpha signal diagnostics</p>

      {/* Controls */}
      <div className="flex items-end gap-3 mb-6">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-1.5" style={{ color: "var(--fg-muted)" }}>
            Alpha Signal
          </div>
          <select value={selected} onChange={(e) => setSelected(e.target.value)}
            style={{
              background: "var(--bg-tertiary)", border: "1px solid var(--border)",
              color: "var(--fg)", borderRadius: 8, padding: "8px 12px", fontSize: 13, cursor: "pointer", minWidth: 240,
            }}>
            {alphas.map((a) => <option key={a.name} value={a.name}>{a.name}</option>)}
          </select>
        </div>
        <button onClick={handleAnalyze} disabled={loading}
          className="flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold"
          style={{
            background: loading ? "var(--bg-tertiary)" : "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
            color: loading ? "var(--fg-muted)" : "#0a0e17",
            border: "none", cursor: loading ? "wait" : "pointer",
            boxShadow: loading ? "none" : "0 0 15px rgba(0, 212, 170, 0.2)",
          }}>
          {loading ? <Loader size={16} className="animate-spin" /> : <FlaskConical size={16} />}
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {/* Alpha info */}
      {info && (
        <div className="text-xs mb-6 p-3.5 rounded-lg font-mono glow-card" style={{ background: "var(--card)" }}>
          <span style={{ color: "var(--fg-muted)" }}>module: </span>
          <span style={{ color: "var(--accent-blue)" }}>{info.module}</span>
          <span className="mx-3" style={{ color: "var(--border-bright)" }}>|</span>
          <span style={{ color: "var(--fg-secondary)" }}>{info.description}</span>
          <span className="mx-3" style={{ color: "var(--border-bright)" }}>|</span>
          <span style={{ color: "var(--fg-muted)" }}>params: </span>
          <span style={{ color: "var(--accent-purple)" }}>
            {Object.entries(info.params).map(([k, v]) => `${k}=${v}`).join(", ")}
          </span>
        </div>
      )}

      {error && (
        <div className="text-sm px-4 py-3 rounded-lg mb-4 font-mono"
          style={{ background: "rgba(255, 71, 87, 0.12)", color: "var(--danger)", border: "1px solid rgba(255, 71, 87, 0.3)" }}>
          {error}
        </div>
      )}

      {analysis && (
        <div className="space-y-5">
          {/* Signal stats */}
          {stats && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <MetricCard label="Mean" value={stats.mean?.toFixed(4) ?? "-"} />
              <MetricCard label="Std Dev" value={stats.std?.toFixed(4) ?? "-"} />
              <MetricCard label="Skewness" value={stats.skew?.toFixed(3) ?? "-"} />
              <MetricCard label="Kurtosis" value={stats.kurtosis?.toFixed(2) ?? "-"} />
            </div>
          )}

          {/* IC stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard label="IC Mean" value={analysis.ic_mean.toFixed(4)} positive={analysis.ic_mean > 0} />
            <MetricCard label="IC Std" value={analysis.ic_std.toFixed(4)} />
            <MetricCard label="IC IR" value={analysis.ic_std > 0 ? (analysis.ic_mean / analysis.ic_std).toFixed(3) : "-"}
              positive={analysis.ic_mean > 0} subtitle="IC Mean / IC Std" />
            <MetricCard label="% Positive" value={stats?.pct_positive !== undefined ? (stats.pct_positive * 100).toFixed(1) + "%" : "-"} />
          </div>

          {/* IC time series */}
          {analysis.ic_series.length > 0 && (
            <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
              <TSLineChart data={analysis.ic_series} title="Information Coefficient (Rank IC)"
                color="#8b5cf6" height={250} referenceLine={0} formatValue={(v) => v.toFixed(3)} />
            </div>
          )}

          {/* Quantile returns */}
          {qr && Object.keys(qr).length > 0 && (
            <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
              <SimpleBarChart
                data={Object.entries(qr).map(([name, value]) => ({ name, value }))}
                title="Annualized Return by Quintile"
                color="#00d4aa"
                formatValue={(v) => (v * 100).toFixed(1) + "%"}
              />
            </div>
          )}

          {/* Signal distribution */}
          {analysis.signal_distribution.length > 0 && (
            <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
              <SimpleBarChart
                data={analysis.signal_distribution.map((p) => ({ name: p.date, value: p.value }))}
                title="Signal Distribution (z-score)"
                color="#3b82f6" height={200}
                formatValue={(v) => v.toFixed(0)}
              />
            </div>
          )}
        </div>
      )}

      {!analysis && !loading && (
        <div className="flex items-center justify-center h-56 rounded-xl grid-bg glow-card"
          style={{ background: "var(--card)" }}>
          <div className="text-center">
            <FlaskConical size={32} style={{ color: "var(--fg-muted)", margin: "0 auto 12px" }} />
            <p className="text-sm" style={{ color: "var(--fg-secondary)" }}>Select an alpha and click Analyze</p>
          </div>
        </div>
      )}
    </div>
  );
}
