import { useEffect, useState } from "react";
import { Play, Plus, Trash2 } from "lucide-react";
import { listAlphas, listUniverses, submitBacktest } from "../lib/api.ts";
import { ProgressBar } from "../components/ProgressBar.tsx";
import { connectProgress } from "../lib/api.ts";
import type { AlphaConfig, AlphaInfo, BacktestRequest, UniverseInfo } from "../types/api.ts";

interface Props {
  onComplete: () => void;
}

interface AlphaRow {
  name: string;
  weight: number;
  params: Record<string, unknown>;
}

const INPUT: React.CSSProperties = {
  background: "var(--bg-tertiary)",
  border: "1px solid var(--border)",
  color: "var(--fg)",
  borderRadius: 8,
  padding: "8px 12px",
  fontSize: 13,
  outline: "none",
  width: "100%",
  fontFamily: "'Inter', sans-serif",
};

const LABEL: React.CSSProperties = { color: "var(--fg-muted)", fontSize: 10, fontWeight: 600, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.08em" };

export function StrategyBuilder({ onComplete }: Props) {
  const [alphas, setAlphas] = useState<AlphaInfo[]>([]);
  const [universes, setUniverses] = useState<UniverseInfo[]>([]);
  const [strategyName, setStrategyName] = useState("My Strategy");
  const [selectedUniverse, setSelectedUniverse] = useState("");
  const [customTickers, setCustomTickers] = useState("");
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [alphaRows, setAlphaRows] = useState<AlphaRow[]>([{ name: "momentum", weight: 1, params: {} }]);
  const [longPct, setLongPct] = useState(0.2);
  const [shortPct, setShortPct] = useState(0.2);
  const [rebalFreq, setRebalFreq] = useState<"daily" | "weekly" | "monthly">("monthly");
  const [commBps, setCommBps] = useState(5);
  const [slipBps, setSlipBps] = useState(5);
  const [maxPos, setMaxPos] = useState(0.1);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    listAlphas().then(setAlphas).catch(() => {});
    listUniverses().then(setUniverses).catch(() => {});
  }, []);

  const addAlpha = () => {
    const available = alphas.find((a) => !alphaRows.some((r) => r.name === a.name));
    if (available) setAlphaRows([...alphaRows, { name: available.name, weight: 1, params: {} }]);
  };

  const removeAlpha = (idx: number) => setAlphaRows(alphaRows.filter((_, i) => i !== idx));

  const updateAlpha = (idx: number, field: keyof AlphaRow, value: unknown) => {
    const rows = [...alphaRows];
    rows[idx] = { ...rows[idx], [field]: value };
    setAlphaRows(rows);
  };

  const getTickers = (): string[] => {
    if (customTickers.trim()) {
      let input = customTickers.trim();
      // Handle stringified list formats: "['AAPL','MSFT']" or '["AAPL","MSFT"]'
      if (input.startsWith("[")) {
        try {
          const parsed = JSON.parse(input.replace(/'/g, '"'));
          if (Array.isArray(parsed)) input = parsed.join(",");
        } catch {
          input = input.replace(/[\[\]'"]/g, "");
        }
      }
      const tickerRe = /^[A-Z0-9.\-]+$/;
      return [...new Set(
        input.split(/[,\s]+/)
          .map((t) => t.trim().replace(/^['"]|['"]$/g, "").trim().toUpperCase())
          .filter((t) => t.length > 0 && tickerRe.test(t))
      )];
    }
    return universes.find((u) => u.name === selectedUniverse)?.tickers ?? [];
  };

  const handleSubmit = async () => {
    const tickers = getTickers();
    if (tickers.length === 0) { setError("Select a universe or enter tickers"); return; }
    if (alphaRows.length === 0) { setError("Add at least one alpha signal"); return; }
    setError(""); setRunning(true); setProgress(0); setStatus("pending");

    const req: BacktestRequest = {
      name: strategyName, tickers,
      start_date: startDate, end_date: endDate,
      alphas: alphaRows.map((r): AlphaConfig => ({ name: r.name, weight: r.weight, params: r.params })),
      long_pct: longPct, short_pct: shortPct,
      rebalance_freq: rebalFreq, commission_bps: commBps, slippage_bps: slipBps, max_position: maxPos,
    };

    try {
      const meta = await submitBacktest(req);
      const ws = connectProgress(meta.run_id, (data) => {
        setProgress(data.progress); setStatus(data.status);
        if (data.status === "completed" || data.status === "failed") {
          ws.close(); setRunning(false);
          if (data.status === "completed") setTimeout(onComplete, 500);
        }
      });
    } catch (e) { setError(String(e)); setRunning(false); }
  };

  return (
    <div className="flex-1 p-6 overflow-y-auto" style={{ background: "var(--bg)" }}>
      <h1 className="text-2xl font-bold gradient-text mb-1">Strategy Builder</h1>
      <p className="text-sm mb-8" style={{ color: "var(--fg-muted)" }}>Configure and run a quantitative backtest</p>

      <div className="max-w-3xl space-y-6">
        {/* Strategy name */}
        <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
          <div style={LABEL}>Strategy Name</div>
          <input value={strategyName} onChange={(e) => setStrategyName(e.target.value)} style={INPUT} />
        </div>

        {/* Universe + Dates */}
        <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
          <div className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: "var(--fg)" }}>
            <div className="w-1 h-4 rounded-full" style={{ background: "var(--accent-blue)" }} /> Universe & Date Range
          </div>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <div style={LABEL}>Universe</div>
              <select value={selectedUniverse} onChange={(e) => setSelectedUniverse(e.target.value)}
                style={{ ...INPUT, cursor: "pointer" }}>
                <option value="">Custom tickers...</option>
                {universes.map((u) => <option key={u.name} value={u.name}>{u.name} ({u.tickers.length})</option>)}
              </select>
            </div>
            <div>
              <div style={LABEL}>Custom Tickers</div>
              <input value={customTickers} onChange={(e) => setCustomTickers(e.target.value)}
                placeholder="AAPL, MSFT, GOOG" style={INPUT} disabled={!!selectedUniverse} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div style={LABEL}>Start Date</div>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} style={INPUT} />
            </div>
            <div>
              <div style={LABEL}>End Date</div>
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} style={INPUT} />
            </div>
          </div>
        </div>

        {/* Alpha signals */}
        <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
          <div className="flex items-center justify-between mb-4">
            <div className="text-sm font-semibold flex items-center gap-2" style={{ color: "var(--fg)" }}>
              <div className="w-1 h-4 rounded-full" style={{ background: "var(--accent-purple)" }} /> Alpha Signals
            </div>
            <button onClick={addAlpha}
              className="flex items-center gap-1 text-xs px-3 py-1.5 rounded-lg font-mono"
              style={{ background: "var(--accent-dim)", color: "var(--accent)", border: "1px solid rgba(0, 212, 170, 0.2)", cursor: "pointer" }}>
              <Plus size={14} /> Add
            </button>
          </div>
          <div className="space-y-2">
            {alphaRows.map((row, idx) => (
              <div key={idx} className="flex items-center gap-3 p-3 rounded-lg"
                style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border)" }}>
                <select value={row.name} onChange={(e) => updateAlpha(idx, "name", e.target.value)}
                  style={{ ...INPUT, width: 200, background: "var(--bg-secondary)" }}>
                  {alphas.map((a) => <option key={a.name} value={a.name}>{a.name}</option>)}
                </select>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono uppercase" style={{ color: "var(--fg-muted)" }}>W:</span>
                  <input type="number" value={row.weight} min={0} max={10} step={0.1}
                    onChange={(e) => updateAlpha(idx, "weight", parseFloat(e.target.value) || 1)}
                    style={{ ...INPUT, width: 60, textAlign: "center", background: "var(--bg-secondary)" }} />
                </div>
                <div className="flex-1 text-[11px] truncate" style={{ color: "var(--fg-muted)" }}>
                  {alphas.find((a) => a.name === row.name)?.description ?? ""}
                </div>
                <button onClick={() => removeAlpha(idx)}
                  style={{ background: "none", border: "none", cursor: "pointer", color: "var(--fg-muted)", padding: 4 }}>
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Portfolio params */}
        <div className="rounded-xl p-5 glow-card" style={{ background: "var(--card)" }}>
          <div className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: "var(--fg)" }}>
            <div className="w-1 h-4 rounded-full" style={{ background: "var(--success)" }} /> Portfolio & Execution
          </div>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div><div style={LABEL}>Long %</div><input type="number" value={longPct} min={0.05} max={0.5} step={0.05} onChange={(e) => setLongPct(parseFloat(e.target.value))} style={INPUT} /></div>
            <div><div style={LABEL}>Short %</div><input type="number" value={shortPct} min={0.05} max={0.5} step={0.05} onChange={(e) => setShortPct(parseFloat(e.target.value))} style={INPUT} /></div>
            <div><div style={LABEL}>Max Position</div><input type="number" value={maxPos} min={0.01} max={1} step={0.01} onChange={(e) => setMaxPos(parseFloat(e.target.value))} style={INPUT} /></div>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div><div style={LABEL}>Rebalance</div>
              <select value={rebalFreq} onChange={(e) => setRebalFreq(e.target.value as "daily" | "weekly" | "monthly")} style={{ ...INPUT, cursor: "pointer" }}>
                <option value="daily">Daily</option><option value="weekly">Weekly</option><option value="monthly">Monthly</option>
              </select>
            </div>
            <div><div style={LABEL}>Commission (bps)</div><input type="number" value={commBps} min={0} max={100} onChange={(e) => setCommBps(parseFloat(e.target.value))} style={INPUT} /></div>
            <div><div style={LABEL}>Slippage (bps)</div><input type="number" value={slipBps} min={0} max={100} onChange={(e) => setSlipBps(parseFloat(e.target.value))} style={INPUT} /></div>
          </div>
        </div>

        {/* Error / Progress / Submit */}
        {error && (
          <div className="text-sm px-4 py-3 rounded-lg font-mono" style={{ background: "rgba(255, 71, 87, 0.12)", color: "var(--danger)", border: "1px solid rgba(255, 71, 87, 0.3)" }}>
            {error}
          </div>
        )}

        {running && (
          <div className="p-5 rounded-xl glow-card animate-pulse-glow" style={{ background: "var(--card)" }}>
            <ProgressBar progress={progress} status={status} />
          </div>
        )}

        <button onClick={handleSubmit} disabled={running}
          className="flex items-center gap-2 px-8 py-3 rounded-lg text-sm font-bold"
          style={{
            background: running ? "var(--bg-tertiary)" : "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
            color: running ? "var(--fg-muted)" : "#0a0e17",
            border: "none", cursor: running ? "not-allowed" : "pointer",
            boxShadow: running ? "none" : "0 0 25px rgba(0, 212, 170, 0.3)",
          }}>
          <Play size={16} />
          {running ? "Running..." : "Run Backtest"}
        </button>
      </div>
    </div>
  );
}
