#!/usr/bin/env python
"""Run an alpha strategy via the backend BacktestRunner and save performance plot."""
import asyncio
import json
from datetime import date
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from gui.backend.models import BacktestRequest, AlphaConfig, RebalanceFreq
from gui.backend.services.backtest_runner import runner


def _load_universe_tickers() -> list[str]:
    """Load tickers from universe.csv."""
    path = Path(__file__).parent / "universe.csv"
    df = pd.read_csv(path)
    return df["ticker"].tolist()


# Long-only multi-factor strategy (matches stock picker approach that got Sharpe ~0.38)
BACKTEST_REQUEST = BacktestRequest(
    name="Multi-Factor Long-Only Strategy",
    tickers=_load_universe_tickers(),
    start_date=date(2022, 1, 1),
    end_date=date(2026, 2, 8),
    alphas=[
        AlphaConfig(name="momentum", params={"lookback": 252, "skip": 21}, weight=0.35),
        AlphaConfig(name="low_volatility", params={"lookback": 126}, weight=0.25),
        AlphaConfig(name="profitability_proxy", params={"lookback": 252}, weight=0.20),
        AlphaConfig(name="trend_strength", params={"lookback": 252}, weight=0.20),
    ],
    long_only=True,
    long_pct=0.2,
    short_pct=0.0,
    rebalance_freq=RebalanceFreq.monthly,
    commission_bps=3.0,
    slippage_bps=3.0,
    max_position=0.08,
)


async def main():
    # Submit backtest (runs in background thread)
    print("Submitting backtest request...")
    run_id = await runner.submit(BACKTEST_REQUEST)
    print(f"Run ID: {run_id}")

    # Poll for completion
    while True:
        meta = runner.get_run(run_id)
        status = meta.status.value
        progress = meta.progress
        print(f"  Status: {status}, progress: {progress:.0%}")
        if status == "completed":
            break
        if status == "failed":
            print(f"  Error: {meta.error or 'Unknown'}")
            return
        await asyncio.sleep(2)

    # Fetch results
    print("Fetching results...")
    result = runner.get_result(run_id)
    result = result.model_dump()

    # Print performance
    m = result["metrics"]
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"  Total Return:        {m['total_return']:.2%}")
    print(f"  Annualized Return:   {m['annualized_return']:.2%}")
    print(f"  Annualized Vol:      {m['annualized_volatility']:.2%}")
    print(f"  Sharpe Ratio:        {m['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio:       {m['sortino_ratio']:.4f}")
    print(f"  Max Drawdown:        {m['max_drawdown']:.2%}")
    print(f"  Calmar Ratio:        {m['calmar_ratio']:.4f}")
    print(f"  Hit Rate:            {m['hit_rate']:.2%}")
    print(f"  Profit Factor:       {m['profit_factor']:.4f}")
    print("=" * 50)

    # Save plot
    out_dir = Path("runs/alpha_backtest_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"alpha_strategy_{run_id}.png"

    equity = [(p["date"], p["value"]) for p in result["equity_curve"]]
    dd = [(p["date"], p["value"]) for p in result["drawdown_series"]]

    dates = [d for d, _ in equity]
    cum_ret = [v for _, v in equity]
    dd_vals = [v for _, v in dd]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(dates, cum_ret, color="#00d4aa", linewidth=1.5)
    ax1.axhline(1.0, color="#4a5568", linestyle="--", linewidth=0.5)
    ax1.set_title(f"{result['name']} — Equity Curve", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Cumulative Return")
    ax1.tick_params(axis="x", rotation=45)
    ax1.tick_params(axis="x", labelsize=7)

    ax2.fill_between(dates, dd_vals, 0, color="#ff4757", alpha=0.4)
    ax2.plot(dates, dd_vals, color="#ff4757", linewidth=0.8)
    ax2.set_title("Drawdown", fontsize=11)
    ax2.set_ylabel("Drawdown")
    ax2.tick_params(axis="x", rotation=45)
    ax2.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")

    # Save JSON summary
    summary_path = out_dir / f"alpha_strategy_{run_id}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "name": result["name"],
                "metrics": m,
                "request": BACKTEST_REQUEST.model_dump() if hasattr(BACKTEST_REQUEST, "model_dump") else str(BACKTEST_REQUEST),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
