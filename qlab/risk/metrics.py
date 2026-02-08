"""Performance and risk metrics.

All functions operate on a simple pandas Series of daily returns
indexed by date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(returns: pd.Series) -> float:
    """Cumulative total return: ``prod(1 + r) - 1``."""
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate (CAGR)."""
    cum = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float(cum ** (1 / n_years) - 1)


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Annualised standard deviation of returns."""
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns : Series
        Daily portfolio returns.
    risk_free_rate : float
        Annual risk-free rate (default 0).
    periods_per_year : int
        Trading days per year.
    """
    excess = returns - risk_free_rate / periods_per_year
    ann_ret = annualized_return(excess, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return float(ann_ret / ann_vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sortino ratio (downside deviation in denominator)."""
    excess = returns - risk_free_rate / periods_per_year
    ann_ret = annualized_return(excess, periods_per_year)
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if ann_ret > 0 else 0.0
    down_vol = float(downside.std() * np.sqrt(periods_per_year))
    if down_vol == 0:
        return 0.0
    return float(ann_ret / down_vol)


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    ann = annualized_return(returns, periods_per_year)
    return float(ann / abs(mdd))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.25 = 25% drawdown)."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1
    return float(dd.min())


def hit_rate(returns: pd.Series) -> float:
    """Fraction of days with positive returns."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def profit_factor(returns: pd.Series) -> float:
    """Sum of positive returns / absolute sum of negative returns."""
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def performance_summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute a full set of performance statistics.

    Returns a dictionary with all major metrics.
    """
    return {
        "total_return": total_return(returns),
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "profit_factor": profit_factor(returns),
        "num_days": len(returns),
    }
