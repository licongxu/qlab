"""Plugin registry for alpha signal discovery."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from qlab.alphas import (
    momentum,
    short_term_reversal,
    trend_strength,
    mean_reversion_zscore,
    rsi_signal,
    bollinger_signal,
    low_volatility,
    idiosyncratic_vol,
    beta_signal,
    profitability_proxy,
    stability,
)


@dataclass
class AlphaEntry:
    name: str
    fn: Callable
    description: str
    module: str
    default_params: dict[str, Any] = field(default_factory=dict)


class AlphaRegistry:
    """Discovers and manages alpha signal functions."""

    def __init__(self) -> None:
        self._registry: dict[str, AlphaEntry] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        builtins = [
            ("momentum", momentum, "12-minus-1 momentum signal",
             {"lookback": 252, "skip": 21}),
            ("short_term_reversal", short_term_reversal,
             "Short-term (1-month) reversal", {"lookback": 21}),
            ("trend_strength", trend_strength,
             "Trend strength (R-squared of price vs time)", {"lookback": 252}),
            ("mean_reversion_zscore", mean_reversion_zscore,
             "Mean reversion z-score", {"lookback": 20}),
            ("rsi_signal", rsi_signal,
             "RSI-based signal (inverted: low RSI = buy)", {"period": 14}),
            ("bollinger_signal", bollinger_signal,
             "Bollinger band z-score signal", {"window": 20, "num_std": 2.0}),
            ("low_volatility", low_volatility,
             "Low volatility anomaly (negative vol rank)",
             {"lookback": 252}),
            ("idiosyncratic_vol", idiosyncratic_vol,
             "Idiosyncratic volatility (needs market_prices)",
             {"lookback": 252}),
            ("beta_signal", beta_signal,
             "Betting-against-beta signal (needs market_prices)",
             {"lookback": 252}),
            ("profitability_proxy", profitability_proxy,
             "Profitability proxy (rolling Sharpe of returns)",
             {"lookback": 252}),
            ("stability", stability,
             "Return stability (negative vol-of-vol)", {"lookback": 252}),
        ]
        for name, fn, desc, params in builtins:
            mod = inspect.getmodule(fn)
            self._registry[name] = AlphaEntry(
                name=name, fn=fn, description=desc,
                module=mod.__name__ if mod else "unknown",
                default_params=params,
            )

    def register(self, name: str, fn: Callable, description: str = "",
                 default_params: dict[str, Any] | None = None) -> None:
        mod = inspect.getmodule(fn)
        self._registry[name] = AlphaEntry(
            name=name, fn=fn, description=description,
            module=mod.__name__ if mod else "custom",
            default_params=default_params or {},
        )

    def get(self, name: str) -> AlphaEntry:
        if name not in self._registry:
            raise KeyError(f"Alpha '{name}' not found. Available: {list(self._registry)}")
        return self._registry[name]

    def list_all(self) -> list[AlphaEntry]:
        return list(self._registry.values())


# Singleton
registry = AlphaRegistry()
