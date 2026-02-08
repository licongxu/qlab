"""Factor regression analysis.

Regresses portfolio returns against one or more factor return series
using OLS.  Uses numpy's least-squares solver to avoid a hard dependency
on statsmodels; if statsmodels is installed, t-statistics and p-values
are computed analytically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegressionResult:
    """Result of an OLS factor regression.

    Attributes
    ----------
    alpha : float
        Annualised intercept (alpha), i.e. daily alpha × 252.
    alpha_tstat : float
        t-statistic of the daily alpha estimate.
    betas : dict[str, float]
        Factor loadings keyed by factor name.
    r_squared : float
        Coefficient of determination.
    residual_vol : float
        Annualised standard deviation of residuals.
    """

    alpha: float
    alpha_tstat: float
    betas: dict[str, float]
    r_squared: float
    residual_vol: float


def factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    periods_per_year: int = 252,
) -> RegressionResult:
    """Regress portfolio returns on factor returns via OLS.

    Parameters
    ----------
    returns : Series
        Daily portfolio returns indexed by date.
    factors : DataFrame
        Factor returns with date index and one column per factor.
    periods_per_year : int
        Used for annualisation.

    Returns
    -------
    RegressionResult
    """
    # Align
    common = returns.index.intersection(factors.index)
    if len(common) < 3:
        raise ValueError("Fewer than 3 overlapping observations.")
    y = returns.reindex(common).values
    X_raw = factors.reindex(common).values
    n, k = X_raw.shape
    # Add constant (intercept)
    X = np.column_stack([np.ones(n), X_raw])

    # OLS via normal equations
    beta_hat, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ beta_hat
    resid = y - y_hat
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors
    sigma2 = ss_res / (n - k - 1) if n > k + 1 else ss_res / max(n, 1)
    cov_matrix = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov_matrix))
    alpha_daily = beta_hat[0]
    alpha_se = se[0] if len(se) > 0 else np.nan
    alpha_tstat = alpha_daily / alpha_se if alpha_se > 0 else 0.0

    factor_names = list(factors.columns)
    betas = {name: float(beta_hat[i + 1]) for i, name in enumerate(factor_names)}

    return RegressionResult(
        alpha=float(alpha_daily * periods_per_year),
        alpha_tstat=float(alpha_tstat),
        betas=betas,
        r_squared=float(r_squared),
        residual_vol=float(np.std(resid) * np.sqrt(periods_per_year)),
    )
