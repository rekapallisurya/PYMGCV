"""AIC and UBRE model selection criteria.

For GAMs, we need to choose the smoothing parameter λ. Two common approaches:

1. AIC (Akaike Information Criterion):
   AIC = N log(RSS/N) + 2 * EDF
   
   where:
   - RSS = residual sum of squares
   - EDF = effective degrees of freedom (trace of hat matrix)
   - N = sample size

2. UBRE (Unbiased Risk Estimator):
   UBRE = RSS/N + 2 * σ² * EDF/N
   
   where σ² is the residual variance.

Both criteria balance fit (RSS) with model complexity (EDF).

Lower AIC/UBRE → better model.

References:
    - Wood, S.N. (2017). GAMs: An Introduction with R.
    - Rippa, S. (1995). An algorithm for selecting a good smoothing parameter.
      SIAM J. Sci. Stat. Comput, 16(4), 790-800.

Module exports:
    - compute_aic: Compute AIC
    - compute_ubre: Compute UBRE
    - aic_ubre_cv: Combined AIC/UBRE cross-validation
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


def compute_aic(
    residuals: np.ndarray,
    edf: float,
    n: Optional[int] = None,
) -> float:
    """Compute AIC for a GAM model.

    AIC = N log(RSS/N) + 2 * EDF

    Args:
        residuals: Residual vector.
        edf: Effective degrees of freedom.
        n: Sample size. If None, uses len(residuals).

    Returns:
        AIC value.
    """
    residuals = np.asarray(residuals)
    if n is None:
        n = len(residuals)
    
    rss = np.sum(residuals ** 2)
    
    # Avoid log(0)
    if rss <= 0:
        return np.inf
    
    aic = n * np.log(rss / n) + 2 * edf
    return aic


def compute_ubre(
    residuals: np.ndarray,
    edf: float,
    sigma2: Optional[float] = None,
    n: Optional[int] = None,
) -> float:
    """Compute UBRE for a GAM model.

    UBRE = RSS/N + 2 * σ² * EDF/N

    Args:
        residuals: Residual vector.
        edf: Effective degrees of freedom.
        sigma2: Residual variance. If None, estimated from residuals.
        n: Sample size. If None, uses len(residuals).

    Returns:
        UBRE value.
    """
    residuals = np.asarray(residuals)
    if n is None:
        n = len(residuals)
    
    rss = np.sum(residuals ** 2)
    
    if sigma2 is None:
        # Estimate from residuals (biased, but common)
        sigma2 = rss / n
    
    ubre = rss / n + 2 * sigma2 * edf / n
    return ubre


def select_lambda_aic(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
    family_name: str = 'gaussian',
) -> Tuple[float, float, np.ndarray]:
    """Select smoothing parameter using AIC.

    For each λ, fit GAM and compute AIC.

    Args:
        y: Response vector.
        X: Design matrix (fixed + smooth effects).
        S: Penalty matrix (block diagonal).
        lambdas: Smoothing parameters to try. If None, uses log-spaced values.
        family_name: Exponential family ('gaussian', 'binomial', 'poisson').

    Returns:
        (optimal_lambda, min_aic, aic_values)
    """
    if lambdas is None:
        lambdas = np.logspace(-3, 3, 20)
    
    aics = []
    edfs = []
    
    for lam in lambdas:
        try:
            # Fit model (simple case: Gaussian, no link function complications)
            XTX = X.T @ X
            XTy = X.T @ y
            
            # Regularized fit
            H = XTX + lam * S
            beta = np.linalg.solve(H, XTy)
            
            # Predictions and residuals
            y_pred = X @ beta
            residuals = y - y_pred
            
            # EDF ≈ trace(X @ inv(XTX + λS) @ XTX)
            try:
                H_inv = np.linalg.inv(H)
                edf = np.trace(X @ H_inv @ XTX)
            except np.linalg.LinAlgError:
                edf = X.shape[1] / 2  # Rough estimate if inversion fails
            
            aic = compute_aic(residuals, edf, n=len(y))
            aics.append(aic)
            edfs.append(edf)
        except:
            aics.append(np.inf)
            edfs.append(np.nan)
    
    aics = np.array(aics)
    best_idx = np.argmin(aics)
    optimal_lambda = lambdas[best_idx]
    min_aic = aics[best_idx]
    
    return optimal_lambda, min_aic, aics


def select_lambda_ubre(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
    family_name: str = 'gaussian',
) -> Tuple[float, float, np.ndarray]:
    """Select smoothing parameter using UBRE.

    Args:
        y: Response vector.
        X: Design matrix.
        S: Penalty matrix.
        lambdas: Smoothing parameters to try.
        family_name: Exponential family name.

    Returns:
        (optimal_lambda, min_ubre, ubre_values)
    """
    if lambdas is None:
        lambdas = np.logspace(-3, 3, 20)
    
    ubres = []
    edfs = []
    
    for lam in lambdas:
        try:
            XTX = X.T @ X
            XTy = X.T @ y
            
            H = XTX + lam * S
            beta = np.linalg.solve(H, XTy)
            
            y_pred = X @ beta
            residuals = y - y_pred
            
            try:
                H_inv = np.linalg.inv(H)
                edf = np.trace(X @ H_inv @ XTX)
            except np.linalg.LinAlgError:
                edf = X.shape[1] / 2
            
            # Estimate sigma2 from residuals
            sigma2 = np.sum(residuals ** 2) / len(y)
            
            ubre = compute_ubre(residuals, edf, sigma2=sigma2, n=len(y))
            ubres.append(ubre)
            edfs.append(edf)
        except:
            ubres.append(np.inf)
            edfs.append(np.nan)
    
    ubres = np.array(ubres)
    best_idx = np.argmin(ubres)
    optimal_lambda = lambdas[best_idx]
    min_ubre = ubres[best_idx]
    
    return optimal_lambda, min_ubre, ubres


def compare_criteria(
    y: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    lambdas: Optional[np.ndarray] = None,
) -> dict:
    """Compare AIC and UBRE criteria for λ selection.

    Args:
        y: Response vector.
        X: Design matrix.
        S: Penalty matrix.
        lambdas: Smoothing parameters to evaluate.

    Returns:
        Dictionary with AIC, UBRE, and optimal λ values.
    """
    lam_aic, aic_min, aics = select_lambda_aic(y, X, S, lambdas)
    lam_ubre, ubre_min, ubres = select_lambda_ubre(y, X, S, lambdas)
    
    return {
        'lambdas': lambdas if lambdas is not None else np.logspace(-3, 3, 20),
        'aic_values': aics,
        'ubre_values': ubres,
        'optimal_lambda_aic': lam_aic,
        'min_aic': aic_min,
        'optimal_lambda_ubre': lam_ubre,
        'min_ubre': ubre_min,
    }
