"""Backtest configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class BacktestConfig:
    """Immutable configuration for a backtest run.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value in dollars.
    commission_bps : float
        Round-trip commission in basis points of traded value.
    slippage_bps : float
        Market-impact / slippage in basis points per unit traded.
    rebalance_freq : str
        ``'daily'``, ``'weekly'``, or ``'monthly'``.
    signal_lag : int
        Number of trading days between signal observation and trade execution.
        Default 1 means: signal observed at close of day *t*, traded at
        *execution_price* of day *t+1*.
    execution_price : str
        Which price to use for fills: ``'open'`` or ``'close'``.
    """

    initial_capital: float = 1_000_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    rebalance_freq: Literal["daily", "weekly", "monthly"] = "monthly"
    signal_lag: int = 1
    execution_price: Literal["open", "close"] = "open"
