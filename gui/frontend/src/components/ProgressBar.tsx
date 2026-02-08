interface Props {
  progress: number;
  status: string;
}

export function ProgressBar({ progress, status }: Props) {
  const pct = Math.max(0, Math.min(100, progress * 100));
  const color = status === "failed" ? "var(--danger)"
    : status === "completed" ? "var(--success)"
    : "var(--accent)";
  const glow = status === "failed" ? "var(--danger)" : "var(--accent)";

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs mb-2 font-mono" style={{ color: "var(--fg-secondary)" }}>
        <span className="uppercase tracking-wider text-[10px]">{status}</span>
        <span>{pct.toFixed(0)}%</span>
      </div>
      <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: "var(--bg-tertiary)" }}>
        <div className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}, var(--accent-blue))`,
            boxShadow: `0 0 10px ${glow}`,
          }} />
      </div>
    </div>
  );
}
