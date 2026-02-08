import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, AreaChart, Area, BarChart, Bar,
  ReferenceLine,
} from "recharts";
import type { TimeSeriesPoint } from "../types/api.ts";

const TOOLTIP_STYLE = {
  background: "#111827",
  border: "1px solid #2a3a5c",
  borderRadius: 10,
  fontSize: 12,
  boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
};

interface LineChartProps {
  data: TimeSeriesPoint[];
  color?: string;
  title?: string;
  height?: number;
  formatValue?: (v: number) => string;
  referenceLine?: number;
}

export function TSLineChart({ data, color = "#00d4aa", title, height = 300, formatValue, referenceLine }: LineChartProps) {
  const fmt = formatValue ?? ((v: number) => v.toFixed(4));
  const sparse = data.length > 500 ? data.filter((_, i) => i % Math.ceil(data.length / 500) === 0) : data;

  return (
    <div>
      {title && (
        <div className="text-sm font-semibold mb-3 flex items-center gap-2" style={{ color: "var(--fg)" }}>
          <div className="w-1 h-4 rounded-full" style={{ background: color }} />
          {title}
        </div>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={sparse} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#4a5568" }}
            tickFormatter={(d: string) => d.slice(0, 7)} interval="preserveStartEnd"
            axisLine={{ stroke: "var(--border)" }} tickLine={{ stroke: "var(--border)" }} />
          <YAxis tick={{ fontSize: 10, fill: "#4a5568" }} tickFormatter={(v: number) => fmt(v)} width={60}
            axisLine={{ stroke: "var(--border)" }} tickLine={{ stroke: "var(--border)" }} />
          <Tooltip contentStyle={TOOLTIP_STYLE}
            formatter={(v: unknown) => [fmt(Number(v)), ""]} labelStyle={{ color: "#8892a8" }} />
          {referenceLine !== undefined && (
            <ReferenceLine y={referenceLine} stroke="#4a5568" strokeDasharray="4 4" strokeWidth={0.8} />
          )}
          <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false}
            style={{ filter: `drop-shadow(0 0 4px ${color}40)` }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TSAreaChart({ data, color = "#ff4757", title, height = 200, formatValue }: LineChartProps) {
  const fmt = formatValue ?? ((v: number) => (v * 100).toFixed(1) + "%");
  const sparse = data.length > 500 ? data.filter((_, i) => i % Math.ceil(data.length / 500) === 0) : data;

  return (
    <div>
      {title && (
        <div className="text-sm font-semibold mb-3 flex items-center gap-2" style={{ color: "var(--fg)" }}>
          <div className="w-1 h-4 rounded-full" style={{ background: color }} />
          {title}
        </div>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={sparse} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <defs>
            <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.3} />
              <stop offset="100%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#4a5568" }}
            tickFormatter={(d: string) => d.slice(0, 7)} interval="preserveStartEnd"
            axisLine={{ stroke: "var(--border)" }} tickLine={{ stroke: "var(--border)" }} />
          <YAxis tick={{ fontSize: 10, fill: "#4a5568" }} tickFormatter={(v: number) => fmt(v)} width={60}
            axisLine={{ stroke: "var(--border)" }} tickLine={{ stroke: "var(--border)" }} />
          <Tooltip contentStyle={TOOLTIP_STYLE}
            formatter={(v: unknown) => [fmt(Number(v)), ""]} />
          <Area type="monotone" dataKey="value" stroke={color} fill="url(#areaGrad)" strokeWidth={1.5} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

interface BarChartProps {
  data: { name: string; value: number }[];
  color?: string;
  title?: string;
  height?: number;
  formatValue?: (v: number) => string;
}

export function SimpleBarChart({ data, color = "#00d4aa", title, height = 250, formatValue }: BarChartProps) {
  const fmt = formatValue ?? ((v: number) => v.toFixed(4));

  return (
    <div>
      {title && (
        <div className="text-sm font-semibold mb-3 flex items-center gap-2" style={{ color: "var(--fg)" }}>
          <div className="w-1 h-4 rounded-full" style={{ background: color }} />
          {title}
        </div>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#8892a8" }}
            axisLine={{ stroke: "var(--border)" }} tickLine={{ stroke: "var(--border)" }} />
          <YAxis tick={{ fontSize: 10, fill: "#4a5568" }} tickFormatter={(v: number) => fmt(v)} width={50}
            axisLine={{ stroke: "var(--border)" }} tickLine={{ stroke: "var(--border)" }} />
          <Tooltip contentStyle={TOOLTIP_STYLE}
            formatter={(v: unknown) => [fmt(Number(v)), ""]} />
          <Bar dataKey="value" fill={color} radius={[4, 4, 0, 0]}
            style={{ filter: `drop-shadow(0 0 4px ${color}30)` }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
