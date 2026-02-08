"""Risk and performance analysis."""

from qlab.risk.metrics import (
    total_return,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    hit_rate,
    profit_factor,
    performance_summary,
)
from qlab.risk.drawdown import (
    drawdown_series,
    drawdown_details,
)
from qlab.risk.regression import (
    factor_regression,
    RegressionResult,
)

__all__ = [
    "total_return", "annualized_return", "annualized_volatility",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "max_drawdown", "hit_rate", "profit_factor", "performance_summary",
    "drawdown_series", "drawdown_details",
    "factor_regression", "RegressionResult",
]
