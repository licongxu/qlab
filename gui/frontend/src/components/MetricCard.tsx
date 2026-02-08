interface Props {
  label: string;
  value: string;
  subtitle?: string;
  positive?: boolean;
}

export function MetricCard({ label, value, subtitle, positive }: Props) {
  const valueColor = positive === undefined
    ? "var(--fg)"
    : positive ? "var(--success)" : "var(--danger)";

  return (
    <div className="rounded-xl p-4 glow-card"
      style={{ background: "var(--card)" }}>
      <div className="text-[11px] font-medium uppercase tracking-wider mb-2"
        style={{ color: "var(--fg-muted)" }}>{label}</div>
      <div className="text-2xl font-bold tracking-tight font-mono"
        style={{ color: valueColor }}>
        {value}
      </div>
      {subtitle && (
        <div className="text-[10px] mt-1.5 font-mono" style={{ color: "var(--fg-muted)" }}>{subtitle}</div>
      )}
    </div>
  );
}
