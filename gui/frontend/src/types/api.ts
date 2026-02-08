export interface AlphaConfig {
  name: string;
  params: Record<string, unknown>;
  weight: number;
}

export interface BacktestRequest {
  name: string;
  tickers: string[];
  start_date: string;
  end_date: string;
  alphas: AlphaConfig[];
  long_only?: boolean;
  long_pct: number;
  short_pct: number;
  rebalance_freq: "daily" | "weekly" | "monthly";
  commission_bps: number;
  slippage_bps: number;
  max_position: number;
}

export interface RunMeta {
  run_id: string;
  name: string;
  status: "pending" | "running" | "completed" | "failed";
  created_at: string;
  tickers: string[];
  start_date: string;
  end_date: string;
  progress: number;
  error: string | null;
}

export interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  hit_rate: number;
  profit_factor: number;
}

export interface TimeSeriesPoint {
  date: string;
  value: number;
}

export interface DrawdownEpisode {
  start: string;
  trough: string;
  end: string | null;
  depth: number;
  days: number;
  recovery_days: number | null;
}

export interface HoldingSnapshot {
  date: string;
  ticker: string;
  weight: number;
}

export interface BacktestResult {
  run_id: string;
  name: string;
  metrics: PerformanceMetrics;
  equity_curve: TimeSeriesPoint[];
  drawdown_series: TimeSeriesPoint[];
  rolling_sharpe: TimeSeriesPoint[];
  monthly_returns: TimeSeriesPoint[];
  drawdown_episodes: DrawdownEpisode[];
  holdings: HoldingSnapshot[];
  turnover: TimeSeriesPoint[];
  gross_exposure: TimeSeriesPoint[];
  net_exposure: TimeSeriesPoint[];
}

export interface AlphaInfo {
  name: string;
  description: string;
  module: string;
  params: Record<string, unknown>;
}

export interface AlphaAnalysis {
  name: string;
  signal_stats: Record<string, number>;
  ic_series: TimeSeriesPoint[];
  ic_mean: number;
  ic_std: number;
  quantile_returns: Record<string, number>;
  signal_distribution: TimeSeriesPoint[];
}

export interface UniverseInfo {
  name: string;
  tickers: string[];
  description: string;
}
