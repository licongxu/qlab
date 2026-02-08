import { BarChart3, FlaskConical, LayoutDashboard, TrendingUp } from "lucide-react";

type Page = "dashboard" | "builder" | "alphalab";

interface Props {
  page: Page;
  onNavigate: (p: Page) => void;
}

const NAV: { id: Page; label: string; icon: typeof LayoutDashboard; shortcut: string }[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard, shortcut: "1" },
  { id: "builder", label: "Strategy Builder", icon: TrendingUp, shortcut: "2" },
  { id: "alphalab", label: "Alpha Lab", icon: FlaskConical, shortcut: "3" },
];

export function Sidebar({ page, onNavigate }: Props) {
  return (
    <aside className="w-60 h-screen flex flex-col shrink-0"
      style={{ background: "var(--bg-secondary)", borderRight: "1px solid var(--border)" }}>
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-6">
        <div className="w-9 h-9 rounded-lg flex items-center justify-center"
          style={{ background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))" }}>
          <BarChart3 size={20} color="#0a0e17" strokeWidth={2.5} />
        </div>
        <div>
          <div className="text-lg font-bold tracking-tight gradient-text">qlab</div>
          <div className="text-[10px] font-mono tracking-widest uppercase" style={{ color: "var(--fg-muted)" }}>
            quant platform
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="mx-4 mb-3" style={{ borderTop: "1px solid var(--border)" }} />

      {/* Nav */}
      <nav className="flex-1 px-3 space-y-1">
        <div className="text-[10px] font-semibold uppercase tracking-[0.15em] px-3 mb-2"
          style={{ color: "var(--fg-muted)" }}>Navigation</div>
        {NAV.map(({ id, label, icon: Icon, shortcut }) => {
          const active = page === id;
          return (
            <button key={id} onClick={() => onNavigate(id)}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm relative group"
              style={{
                background: active ? "var(--accent-dim)" : "transparent",
                color: active ? "var(--accent)" : "var(--fg-secondary)",
                fontWeight: active ? 600 : 400,
                border: active ? "1px solid rgba(0, 212, 170, 0.2)" : "1px solid transparent",
                cursor: "pointer",
              }}>
              {active && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full"
                  style={{ background: "var(--accent)" }} />
              )}
              <Icon size={18} />
              <span className="flex-1 text-left">{label}</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                style={{
                  background: active ? "rgba(0, 212, 170, 0.15)" : "var(--bg-tertiary)",
                  color: active ? "var(--accent)" : "var(--fg-muted)",
                }}>
                {shortcut}
              </span>
            </button>
          );
        })}
      </nav>

      {/* Status bar */}
      <div className="px-5 py-4" style={{ borderTop: "1px solid var(--border)" }}>
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full" style={{ background: "var(--success)", boxShadow: "0 0 6px var(--success)" }} />
          <span className="text-xs" style={{ color: "var(--fg-muted)" }}>System Online</span>
        </div>
        <div className="text-[10px] font-mono" style={{ color: "var(--fg-muted)" }}>
          v0.1.0 &middot; {new Date().toLocaleDateString()}
        </div>
      </div>
    </aside>
  );
}
