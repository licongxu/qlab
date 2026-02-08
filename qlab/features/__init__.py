"""Feature engineering: returns, volatility, rolling stats, cross-sectional transforms."""

from qlab.features.returns import (
    simple_returns,
    log_returns,
    cumulative_returns,
    excess_returns,
)
from qlab.features.volatility import (
    realized_volatility,
    ewm_volatility,
    parkinson_volatility,
    garman_klass_volatility,
)
from qlab.features.rolling import (
    rolling_mean,
    rolling_std,
    rolling_skew,
    rolling_kurt,
    rolling_correlation,
    rolling_beta,
    ewm_mean,
)
from qlab.features.cross_section import (
    rank,
    zscore,
    demean,
    winsorize,
    neutralize,
)

__all__ = [
    "simple_returns", "log_returns", "cumulative_returns", "excess_returns",
    "realized_volatility", "ewm_volatility", "parkinson_volatility",
    "garman_klass_volatility",
    "rolling_mean", "rolling_std", "rolling_skew", "rolling_kurt",
    "rolling_correlation", "rolling_beta", "ewm_mean",
    "rank", "zscore", "demean", "winsorize", "neutralize",
]
