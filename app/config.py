"""Selection pipeline configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SelectionConfig:
    # Universe
    universe_file: str = "universe.csv"
    start_date: str = "2022-01-01"
    end_date: str = "2026-02-08"
    cache_dir: str = ".qlab_cache"

    # Alpha weights (ensemble)
    alpha_weights: dict[str, float] = field(default_factory=lambda: {
        "momentum": 0.30,
        "short_term_reversal": 0.10,
        "mean_reversion_zscore": 0.15,
        "low_volatility": 0.20,
        "profitability_proxy": 0.15,
        "trend_strength": 0.10,
    })

    # Alpha params
    momentum_lookback: int = 252
    momentum_skip: int = 21
    reversal_lookback: int = 21
    mr_lookback: int = 20
    vol_lookback: int = 126
    quality_lookback: int = 252
    trend_lookback: int = 252

    # Portfolio construction
    long_only: bool = True          # For practical usage
    n_long: int = 20                # Top N stocks to select
    n_short: int = 0                # 0 = long-only
    max_position_weight: float = 0.08
    sector_cap: float = 0.30        # Max weight per sector

    # Filters
    min_dollar_volume_20d: float = 5e6       # Min avg daily dollar volume
    max_volatility_annualized: float = 0.80  # Exclude extreme vol stocks
    min_price: float = 5.0                   # Penny stock filter

    # Regime
    regime_filter: bool = True
    regime_vol_lookback: int = 63
    regime_vol_threshold: float = 0.25  # High vol regime threshold (annualized)

    # Backtest
    rebalance_freq: str = "weekly"
    commission_bps: float = 3.0
    slippage_bps: float = 3.0

    # Output
    output_dir: str = "runs"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
