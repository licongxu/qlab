"""Core stock selection pipeline."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qlab.data import ParquetCache, YFinanceProvider
from qlab.features import simple_returns, realized_volatility, rank, zscore, winsorize
from qlab.features.rolling import rolling_mean
from qlab.alphas import (
    momentum, short_term_reversal, trend_strength,
    mean_reversion_zscore, low_volatility, profitability_proxy,
)
from qlab.portfolio import equal_weight_long_short, normalize_weights, apply_position_limits
from qlab.backtest import BacktestConfig, run_backtest
from qlab.risk import performance_summary, drawdown_series, drawdown_details

from .config import SelectionConfig


class StockPicker:
    """Rule-based stock selection engine using qlab alpha signals."""

    def __init__(self, config: SelectionConfig | None = None):
        self.config = config or SelectionConfig()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.run_dir = Path(self.config.output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)

    def run(self) -> dict[str, Any]:
        """Execute the full stock selection pipeline."""
        print(f"[{self.run_id}] Starting stock selection pipeline...")

        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)

        # 1. Load universe
        universe = self._load_universe()
        tickers = universe["ticker"].tolist()
        sector_map = dict(zip(universe["ticker"], universe["sector"]))
        print(f"  Universe: {len(tickers)} tickers")

        # 2. Fetch data
        prices, close, volume = self._fetch_data(tickers)

        # 3. Apply filters
        filtered_tickers, filter_log = self._apply_filters(close, volume, prices)
        removed = set(tickers) - set(filtered_tickers)
        print(f"  After filters: {len(filtered_tickers)} tickers ({len(removed)} removed)")
        if removed:
            for entry in filter_log:
                if entry["ticker"] in removed:
                    print(f"    REMOVED: {entry['ticker']} — {entry['filter']} (value={entry['value']})")

        # 4. Compute alpha signals
        signals, signal_details = self._compute_signals(close, filtered_tickers)
        print(f"  Computed {len(signals)} alpha signals")

        # 5. Combine into composite score
        composite = self._combine_signals(signals)
        print(f"  Composite signal: {len(composite.dropna())} observations")

        # 6. Market regime check
        regime_info = self._check_regime(close)
        print(f"  Market regime: {regime_info['regime']} (vol={regime_info['current_vol']:.1%})")

        # 7. Generate selection for latest date
        selection = self._generate_selection(composite, sector_map, regime_info)
        print(f"  Selected {len(selection)} stocks")

        # 8. Run backtest on historical data
        backtest_summary = self._run_backtest(composite, prices)
        print(f"  Backtest Sharpe: {backtest_summary.get('sharpe_ratio', 0):.3f}")

        # 9. Generate plots
        self._generate_plots(composite, prices, selection)

        # 10. Save outputs
        self._save_outputs(selection, backtest_summary, signal_details, filter_log, regime_info)

        print(f"  Outputs saved to: {self.run_dir}")
        return {
            "run_id": self.run_id,
            "n_selected": len(selection),
            "sharpe": backtest_summary.get("sharpe_ratio", 0),
            "regime": regime_info["regime"],
            "output_dir": str(self.run_dir),
        }

    def _load_universe(self) -> pd.DataFrame:
        path = Path(self.config.universe_file)
        if not path.exists():
            path = Path(__file__).parent.parent / self.config.universe_file
        return pd.read_csv(path)

    def _fetch_data(self, tickers: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        provider = ParquetCache(YFinanceProvider(), cache_dir=self.config.cache_dir)
        prices = provider.fetch(tickers, self.config.start_date, self.config.end_date)
        close = prices["adj_close"]
        volume = prices["volume"]
        return prices, close, volume

    def _apply_filters(
        self, close: pd.Series, volume: pd.Series, prices: pd.DataFrame
    ) -> tuple[list[str], list[dict]]:
        cfg = self.config
        log: list[dict] = []
        latest_date = close.index.get_level_values("date").max()
        all_tickers = close.index.get_level_values("ticker").unique().tolist()
        keep = set(all_tickers)

        for t in all_tickers:
            try:
                t_close = close.xs(t, level="ticker")
                t_vol = volume.xs(t, level="ticker")
            except KeyError:
                keep.discard(t)
                log.append({"ticker": t, "filter": "no_data", "value": 0})
                continue

            # Data completeness
            if len(t_close) < 200:
                keep.discard(t)
                log.append({"ticker": t, "filter": "insufficient_data", "value": len(t_close)})
                continue

            # Price filter
            last_price = float(t_close.iloc[-1])
            if last_price < cfg.min_price:
                keep.discard(t)
                log.append({"ticker": t, "filter": "min_price", "value": last_price})
                continue

            # Dollar volume filter
            dv = float((t_close * t_vol).tail(20).mean())
            if dv < cfg.min_dollar_volume_20d:
                keep.discard(t)
                log.append({"ticker": t, "filter": "min_dollar_volume", "value": dv})
                continue

            # Volatility filter
            if len(t_close) >= 63:
                ret = t_close.pct_change().dropna()
                ann_vol = float(ret.tail(63).std() * np.sqrt(252))
                if ann_vol > cfg.max_volatility_annualized:
                    keep.discard(t)
                    log.append({"ticker": t, "filter": "max_volatility", "value": ann_vol})
                    continue

        return sorted(keep), log

    def _compute_signals(
        self, close: pd.Series, tickers: list[str]
    ) -> tuple[dict[str, pd.Series], pd.DataFrame]:
        cfg = self.config
        # Filter close to selected tickers
        mask = close.index.get_level_values("ticker").isin(tickers)
        c = close[mask]

        alpha_fns = {
            "momentum": lambda: momentum(c, lookback=cfg.momentum_lookback, skip=cfg.momentum_skip),
            "short_term_reversal": lambda: short_term_reversal(c, lookback=cfg.reversal_lookback),
            "mean_reversion_zscore": lambda: mean_reversion_zscore(c, lookback=cfg.mr_lookback),
            "low_volatility": lambda: low_volatility(c, lookback=cfg.vol_lookback),
            "profitability_proxy": lambda: profitability_proxy(c, lookback=cfg.quality_lookback),
            "trend_strength": lambda: trend_strength(c, lookback=cfg.trend_lookback),
        }

        signals = {}
        for name in cfg.alpha_weights:
            if name not in alpha_fns:
                continue
            try:
                raw = alpha_fns[name]()
                raw = raw.dropna()
                if len(raw) == 0:
                    continue
                # Ensure proper MultiIndex
                if not isinstance(raw.index, pd.MultiIndex):
                    continue
                win = winsorize(raw, lower=0.05, upper=0.95)
                normed = zscore(win)
                signals[name] = normed
                print(f"    {name}: {len(normed)} obs")
            except Exception as e:
                print(f"    {name}: FAILED ({e})")

        # Signal details for latest date
        latest_date = c.index.get_level_values("date").max()
        rows = []
        for t in tickers:
            row = {"ticker": t}
            for name, sig in signals.items():
                if (latest_date, t) in sig.index:
                    row[f"signal_{name}"] = round(float(sig.loc[(latest_date, t)]), 4)
                else:
                    row[f"signal_{name}"] = np.nan
            rows.append(row)

        detail_df = pd.DataFrame(rows).set_index("ticker")
        return signals, detail_df

    def _combine_signals(self, signals: dict[str, pd.Series]) -> pd.Series:
        cfg = self.config
        weighted = []
        for name, sig in signals.items():
            w = cfg.alpha_weights.get(name, 0)
            if w > 0:
                weighted.append(sig * w)

        if not weighted:
            raise ValueError("No alpha signals with positive weights")

        # Align on common index
        common = weighted[0].index
        for s in weighted[1:]:
            common = common.intersection(s.index)

        aligned = [s.reindex(common) for s in weighted]
        composite = sum(aligned)
        return zscore(composite.dropna())

    def _check_regime(self, close: pd.Series) -> dict:
        cfg = self.config
        # Compute market-level volatility
        mkt_ret = simple_returns(close).groupby(level="date").mean().dropna()
        trailing_vol = mkt_ret.rolling(cfg.regime_vol_lookback).std() * np.sqrt(252)
        current_vol = float(trailing_vol.iloc[-1])

        if current_vol > cfg.regime_vol_threshold:
            regime = "HIGH_VOL"
            adjustment = 0.7  # Reduce exposure in high vol
        else:
            regime = "NORMAL"
            adjustment = 1.0

        return {
            "regime": regime,
            "current_vol": current_vol,
            "threshold": cfg.regime_vol_threshold,
            "exposure_adjustment": adjustment,
            "trailing_vol_series": trailing_vol.tail(252).to_dict(),
        }

    def _generate_selection(
        self, composite: pd.Series, sector_map: dict[str, str], regime_info: dict
    ) -> pd.DataFrame:
        cfg = self.config
        latest_date = composite.index.get_level_values("date").max()

        # Get latest cross-section
        try:
            latest = composite.loc[latest_date].sort_values(ascending=False)
        except KeyError:
            # Try second-latest date
            dates = composite.index.get_level_values("date").unique()
            latest = composite.loc[dates[-2]].sort_values(ascending=False)
            latest_date = dates[-2]

        # Rank all
        ranks = latest.rank(ascending=False).astype(int)

        if cfg.long_only:
            # Select top N with sector cap enforcement
            selected = []
            sector_counts: dict[str, float] = {}
            max_per_sector = max(1, int(cfg.n_long * cfg.sector_cap))

            for ticker in latest.index:
                if len(selected) >= cfg.n_long:
                    break
                sector = sector_map.get(ticker, "Unknown")
                count = sector_counts.get(sector, 0)
                if count >= max_per_sector:
                    continue
                selected.append(ticker)
                sector_counts[sector] = count + 1

            # Build output
            rows = []
            weight_per_stock = min(cfg.max_position_weight, 1.0 / len(selected)) if selected else 0
            # Adjust for regime
            weight_per_stock *= regime_info["exposure_adjustment"]

            for t in selected:
                sector = sector_map.get(t, "Unknown")
                reasons = []
                score = float(latest[t])
                if score > 1.0:
                    reasons.append("strong_composite")
                elif score > 0.5:
                    reasons.append("above_avg_composite")
                reasons.append(f"rank_{int(ranks[t])}_of_{len(ranks)}")

                rows.append({
                    "ticker": t,
                    "score": round(score, 4),
                    "rank": int(ranks[t]),
                    "weight": round(weight_per_stock, 4),
                    "sector": sector,
                    "reason_codes": "|".join(reasons),
                    "date": str(latest_date.date()),
                })

            df = pd.DataFrame(rows)
        else:
            # Long-short: top N long, bottom N short
            long_tickers = latest.head(cfg.n_long).index.tolist()
            short_tickers = latest.tail(cfg.n_short).index.tolist() if cfg.n_short > 0 else []
            rows = []
            for t in long_tickers:
                rows.append({
                    "ticker": t, "score": round(float(latest[t]), 4),
                    "rank": int(ranks[t]),
                    "weight": round(1.0 / cfg.n_long * regime_info["exposure_adjustment"], 4),
                    "sector": sector_map.get(t, "Unknown"),
                    "side": "LONG",
                    "reason_codes": f"rank_{int(ranks[t])}_of_{len(ranks)}",
                    "date": str(latest_date.date()),
                })
            for t in short_tickers:
                rows.append({
                    "ticker": t, "score": round(float(latest[t]), 4),
                    "rank": int(ranks[t]),
                    "weight": round(-1.0 / cfg.n_short * regime_info["exposure_adjustment"], 4),
                    "sector": sector_map.get(t, "Unknown"),
                    "side": "SHORT",
                    "reason_codes": f"rank_{int(ranks[t])}_of_{len(ranks)}",
                    "date": str(latest_date.date()),
                })
            df = pd.DataFrame(rows)

        return df

    def _run_backtest(self, composite: pd.Series, prices: pd.DataFrame) -> dict:
        cfg = self.config
        try:
            if cfg.long_only:
                # Use proportional weights for long-only
                from qlab.portfolio import proportional_weights
                weights = proportional_weights(composite, long_only=True)
                weights = apply_position_limits(weights, max_weight=cfg.max_position_weight, min_weight=0.0)
                # Renormalize to sum to 1
                gsum = weights.groupby(level="date").sum()
                weights = weights / weights.groupby(level="date").transform("sum")
                weights = weights.fillna(0)
            else:
                weights = equal_weight_long_short(composite, long_pct=0.2, short_pct=0.2)
                weights = normalize_weights(weights, gross_exposure=2.0, net_exposure=0.0)
                weights = apply_position_limits(weights, max_weight=cfg.max_position_weight,
                                                min_weight=-cfg.max_position_weight)

            bt_config = BacktestConfig(
                rebalance_freq=cfg.rebalance_freq,
                commission_bps=cfg.commission_bps,
                slippage_bps=cfg.slippage_bps,
                signal_lag=1,
                execution_price="open",
            )
            result = run_backtest(weights, prices, config=bt_config)
            summary = performance_summary(result.portfolio_returns)

            # Save the backtest result for plotting
            self._bt_result = result
            return summary
        except Exception as e:
            print(f"  Backtest warning: {e}")
            return {"error": str(e)}

    def _generate_plots(self, composite: pd.Series, prices: pd.DataFrame, selection: pd.DataFrame):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.rcParams.update({
                "figure.facecolor": "#0a0e17",
                "axes.facecolor": "#111827",
                "axes.edgecolor": "#1e2a42",
                "text.color": "#e8ecf4",
                "axes.labelcolor": "#8892a8",
                "xtick.color": "#4a5568",
                "ytick.color": "#4a5568",
                "grid.color": "#1e2a42",
                "grid.alpha": 0.5,
                "axes.grid": True,
                "figure.dpi": 150,
                "font.size": 10,
            })

            if hasattr(self, "_bt_result"):
                result = self._bt_result

                # Equity curve + drawdown
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
                cum_ret = (1 + result.portfolio_returns).cumprod()
                ax1.plot(cum_ret.index, cum_ret.values, color="#00d4aa", linewidth=1.5)
                ax1.axhline(1.0, color="#4a5568", linestyle="--", linewidth=0.5)
                ax1.set_title("Equity Curve", fontsize=14, fontweight="bold", color="#e8ecf4")
                ax1.set_ylabel("Cumulative Return")

                dd = drawdown_series(result.portfolio_returns)
                ax2.fill_between(dd.index, dd.values, 0, color="#ff4757", alpha=0.4)
                ax2.plot(dd.index, dd.values, color="#ff4757", linewidth=0.8)
                ax2.set_title("Drawdown", fontsize=11, color="#e8ecf4")
                ax2.set_ylabel("Drawdown")

                plt.tight_layout()
                fig.savefig(self.run_dir / "plots" / "equity_curve.png", bbox_inches="tight")
                plt.close(fig)

                # Turnover
                fig, ax = plt.subplots(figsize=(14, 4))
                ax.bar(result.turnover.index, result.turnover.values, color="#3b82f6", alpha=0.7, width=2)
                ax.set_title("Portfolio Turnover", fontsize=14, fontweight="bold", color="#e8ecf4")
                ax.set_ylabel("Turnover")
                plt.tight_layout()
                fig.savefig(self.run_dir / "plots" / "turnover.png", bbox_inches="tight")
                plt.close(fig)

            # Selection bar chart
            if len(selection) > 0:
                fig, ax = plt.subplots(figsize=(14, 6))
                sel = selection.sort_values("score", ascending=True)
                colors = ["#00d4aa" if s > 0 else "#ff4757" for s in sel["score"]]
                ax.barh(sel["ticker"], sel["score"], color=colors)
                ax.set_title("Selected Stocks — Composite Score", fontsize=14, fontweight="bold", color="#e8ecf4")
                ax.set_xlabel("Composite Alpha Score")
                plt.tight_layout()
                fig.savefig(self.run_dir / "plots" / "selection_scores.png", bbox_inches="tight")
                plt.close(fig)

            print("  Plots saved.")
        except Exception as e:
            print(f"  Plot warning: {e}")

    def _save_outputs(
        self, selection: pd.DataFrame, backtest: dict,
        signal_details: pd.DataFrame, filter_log: list[dict],
        regime_info: dict,
    ):
        # Selection CSV
        selection.to_csv(self.run_dir / "selection.csv", index=False)

        # Signal details
        signal_details.to_csv(self.run_dir / "signal_details.csv")

        # Backtest summary
        bt_clean = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in backtest.items()}
        with open(self.run_dir / "backtest_summary.json", "w") as f:
            json.dump(bt_clean, f, indent=2, default=str)

        # Filter log
        with open(self.run_dir / "filter_log.json", "w") as f:
            json.dump(filter_log, f, indent=2, default=str)

        # Regime info (without large series)
        regime_save = {k: v for k, v in regime_info.items() if k != "trailing_vol_series"}
        with open(self.run_dir / "regime.json", "w") as f:
            json.dump(regime_save, f, indent=2, default=str)
