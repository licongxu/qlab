"""CLI entrypoint for stock selection pipeline.

Usage:
    python -m app.run_selection
    python -m app.run_selection --n-long 30 --long-only
    python -m app.run_selection --start-date 2021-01-01 --rebalance-freq monthly
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import SelectionConfig
from app.stock_picker import StockPicker


def main():
    parser = argparse.ArgumentParser(description="qlab Stock Selection Pipeline")
    parser.add_argument("--universe", default="universe.csv", help="Path to universe CSV")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-02-08")
    parser.add_argument("--n-long", type=int, default=20, help="Number of stocks to select")
    parser.add_argument("--long-only", action="store_true", default=True)
    parser.add_argument("--long-short", action="store_true", help="Enable long-short mode")
    parser.add_argument("--n-short", type=int, default=10)
    parser.add_argument("--max-position", type=float, default=0.08)
    parser.add_argument("--sector-cap", type=float, default=0.30)
    parser.add_argument("--rebalance-freq", default="weekly", choices=["daily", "weekly", "monthly"])
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--no-regime-filter", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()

    config = SelectionConfig(
        universe_file=args.universe,
        start_date=args.start_date,
        end_date=args.end_date,
        n_long=args.n_long,
        long_only=not args.long_short,
        n_short=args.n_short if args.long_short else 0,
        max_position_weight=args.max_position,
        sector_cap=args.sector_cap,
        rebalance_freq=args.rebalance_freq,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        regime_filter=not args.no_regime_filter,
        output_dir=args.output_dir,
    )

    picker = StockPicker(config)
    result = picker.run()

    print("\n" + "=" * 60)
    print("STOCK SELECTION COMPLETE")
    print("=" * 60)
    print(f"  Run ID:    {result['run_id']}")
    print(f"  Selected:  {result['n_selected']} stocks")
    print(f"  Regime:    {result['regime']}")
    print(f"  Backtest Sharpe: {result['sharpe']:.3f}")
    print(f"  Output:    {result['output_dir']}")
    print("=" * 60)

    # Print selection table
    import pandas as pd
    sel = pd.read_csv(Path(result["output_dir"]) / "selection.csv")
    print("\nSELECTED STOCKS (ranked by composite alpha score):")
    print("-" * 80)
    for _, row in sel.iterrows():
        print(f"  {row['rank']:>3d}. {row['ticker']:<6s}  score={row['score']:>+7.4f}  "
              f"weight={row['weight']:.4f}  sector={row['sector']:<25s}")
    print("-" * 80)
    print(f"  Total weight: {sel['weight'].sum():.4f}")


if __name__ == "__main__":
    main()
