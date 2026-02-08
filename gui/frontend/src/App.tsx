import { useState, useMemo } from "react";
import { Sidebar } from "./components/Sidebar.tsx";
import { Dashboard } from "./pages/Dashboard.tsx";
import { StrategyBuilder } from "./pages/StrategyBuilder.tsx";
import { AlphaLab } from "./pages/AlphaLab.tsx";
import { useKeyboard } from "./hooks/useKeyboard.ts";

type Page = "dashboard" | "builder" | "alphalab";

function App() {
  const [page, setPage] = useState<Page>("dashboard");

  const bindings = useMemo(() => ({
    "mod+1": () => setPage("dashboard"),
    "mod+2": () => setPage("builder"),
    "mod+3": () => setPage("alphalab"),
  }), []);

  useKeyboard(bindings);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar page={page} onNavigate={setPage} />
      {page === "dashboard" && <Dashboard onNavigateBuilder={() => setPage("builder")} />}
      {page === "builder" && <StrategyBuilder onComplete={() => setPage("dashboard")} />}
      {page === "alphalab" && <AlphaLab />}
    </div>
  );
}

export default App;
