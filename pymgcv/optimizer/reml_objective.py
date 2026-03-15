"""REML objective for smoothing parameter selection.

Restricted Maximum Likelihood (REML) criterion for choosing smoothing parameters.

REML = -1/2 [ log|X^T W X + Sλ| + y^T P y ]

where P = (X^T W X + Sλ)⁻¹ is the precision matrix.

Gradients:
    ∂REML/∂λⱼ = -1/2 [ trace((X^T W X + Sλ)⁻¹ Sⱼ) + y^T P Sⱼ P y ]

References:
    - Wood, S. N. (2011): Fast stable restricted max likelihood
    - Wood, S. N. (2017): Generalized Additive Models

Module exports:
    - REMLObjective: Main class
    - compute_reml: Functional API
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg


class REMLObjective:
    """Restricted Maximum Likelihood objective for smoothing parameters.

    Computes REML score and its derivatives w.r.t. λⱼ.

    Attributes:
        X: Design matrix.
        y: Response.
        family: Distribution family.
        S_list: List of penalty matrices.
        smooth_starts: Starting column for each smooth term.
        smooth_sizes: Basis dimension for each smooth term.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        family: object,
        S_list: list[np.ndarray],
        smooth_starts: list[int],
        smooth_sizes: list[int],
        offset: Optional[np.ndarray] = None,
        dispersion: float = 1.0,
    ) -> None:
        """Initialize REML objective.

        Args:
            X: Design matrix, shape (n, p).
            y: Response, shape (n,).
            family: Distribution family.
            S_list: Penalty matrices.
            smooth_starts: Starting column index for each smooth.
            smooth_sizes: Basis dimension for each smooth.
            offset: Offset vector.
            dispersion: Dispersion parameter φ.
        """
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.family = family
        self.S_list = [np.asarray(S, dtype=np.float64) for S in S_list]
        self.smooth_starts = smooth_starts
        self.smooth_sizes = smooth_sizes
        self.offset = np.asarray(offset, dtype=np.float64) if offset is not None else np.zeros_like(y)
        self.dispersion = float(dispersion)

        self.n, self.p = self.X.shape
        self.n_smooth = len(S_list)

    def objective(self, beta: np.ndarray, log_lambda: np.ndarray) -> float:
        r"""Compute REML score.

        REML = -1/2 [ log|X^T W X + Sλ| + y^T P y ]

        Args:
            beta: Fitted coefficients at current λ.
            log_lambda: Log of smoothing parameters (log λⱼ).

        Returns:
            REML score (scalar).
        """
        lambda_vec = np.exp(log_lambda)

        # Construct combined penalty
        S_combined = self._construct_combined_penalty(lambda_vec)

        # GLM weight matrix at current beta
        eta = self.X @ beta + self.offset
        mu = self.family.linkinv(eta)
        dmu_deta = self.family.dmu_deta(eta)
        var_mu = self.family.variance(mu, self.dispersion)
        
        w = (dmu_deta**2) / var_mu
        
        # Construct X^T W X + S
        XtWX = self.X.T @ (self.X * w[:, np.newaxis])
        A = XtWX + S_combined

        # Log determinant of A
        try:
            sign, logdet = linalg.slogdet(A)
            logdet_A = logdet if sign > 0 else np.inf
        except:
            logdet_A = np.inf

        # Precision (inverse of A)
        try:
            P = linalg.inv(A)
        except linalg.LinAlgError:
            return np.inf

        # Quadratic form: y^T P y
        roots = P @ self.y
        quad_form = np.sum(self.y * roots)

        # REML
        reml = -0.5 * (logdet_A + quad_form)

        return reml

    def gradient_wrt_log_lambda(
        self, beta: np.ndarray, log_lambda: np.ndarray
    ) -> np.ndarray:
        r"""Compute gradient of REML w.r.t. log(λ).

        Uses chain rule:
            ∂REML/∂(log λⱼ) = ∂REML/∂λⱼ * λⱼ

        where:
            ∂REML/∂λⱼ = -1/2 [ trace(P Sⱼ) + (Xβ)^T P Sⱼ P (Xβ) ]

        Args:
            beta: Fitted coefficients.
            log_lambda: Log smoothing parameters.

        Returns:
            Gradient vector, shape (n_smooth,).
        """
        try:
            lambda_vec = np.exp(log_lambda)

            S_combined = self._construct_combined_penalty(lambda_vec)

            eta = self.X @ beta + self.offset
            mu = self.family.linkinv(eta)
            dmu_deta = self.family.dmu_deta(eta)
            var_mu = np.maximum(self.family.variance(mu, self.dispersion), 1e-10)

            w = np.clip((dmu_deta ** 2) / var_mu, 0, 1e10)

            XtWX = self.X.T @ (self.X * w[:, np.newaxis])
            A = XtWX + S_combined + 1e-6 * np.eye(self.p)  # Ridge for stability

            P = linalg.pinv(A)  # pseudo-inverse handles near-singular A

            # Use fitted linear predictor (p-dimensional) not y (n-dimensional)
            Xbeta = self.X @ beta  # (n,) projected version via X^T
            Px = P @ (self.X.T @ (w * Xbeta))  # (p,)

            grad_log_lambda = np.zeros(self.n_smooth)

            for j, S_j in enumerate(self.S_list):
                PS_j = P @ S_j
                trace_PSj = np.trace(PS_j)
                quad_term = Xbeta @ (self.X @ (PS_j @ (self.X.T @ (w * Xbeta))))
                grad_lambdaj = -0.5 * (trace_PSj + quad_term)
                grad_log_lambda[j] = grad_lambdaj * lambda_vec[j]

            return grad_log_lambda
        except Exception:
            return np.full(self.n_smooth, np.nan)

    def hessian_wrt_log_lambda(
        self, beta: np.ndarray, log_lambda: np.ndarray
    ) -> np.ndarray:
        r"""Compute Hessian of REML w.r.t. log(λ).

        Hessian[j,k] = ∂²REML / (∂(log λⱼ) ∂(log λₖ))

        Args:
            beta: Fitted coefficients.
            log_lambda: Log smoothing parameters.

        Returns:
            Hessian matrix, shape (n_smooth, n_smooth).
        """
        try:
            lambda_vec = np.exp(log_lambda)

            S_combined = self._construct_combined_penalty(lambda_vec)

            eta = self.X @ beta + self.offset
            mu = self.family.linkinv(eta)
            dmu_deta = self.family.dmu_deta(eta)
            var_mu = np.maximum(self.family.variance(mu, self.dispersion), 1e-10)

            w = np.clip((dmu_deta ** 2) / var_mu, 0, 1e10)

            XtWX = self.X.T @ (self.X * w[:, np.newaxis])
            A = XtWX + S_combined + 1e-6 * np.eye(self.p)  # Ridge for stability

            P = linalg.pinv(A)

            # Use fitted values instead of raw y to avoid dimension mismatch
            Xbeta = self.X @ beta  # (n,)
            Py_proxy = P @ (self.X.T @ (w * Xbeta))  # (p,) coefficient-space proxy

            H = np.zeros((self.n_smooth, self.n_smooth))

            for j in range(self.n_smooth):
                for k in range(j, self.n_smooth):
                    S_j = self.S_list[j]
                    S_k = self.S_list[k]

                    PS_j = P @ S_j
                    PS_k = P @ S_k

                    trace1 = np.trace(PS_j @ PS_k)

                    PSk_Py = P @ (S_k @ Py_proxy)
                    quad1 = Py_proxy @ (S_j @ PSk_Py)

                    h_jk = 0.5 * (trace1 + quad1) * lambda_vec[j] * lambda_vec[k]

                    H[j, k] = h_jk
                    H[k, j] = h_jk

            return H
        except Exception:
            return np.full((self.n_smooth, self.n_smooth), np.nan)
        for j in range(self.n_smooth):
            for k in range(j, self.n_smooth):
                S_j = self.S_list[j]
                S_k = self.S_list[k]
                
                PS_j = P @ S_j
                PS_k = P @ S_k
                
                # Trace terms
                trace1 = np.trace(PS_j @ PS_k)
                
                # Quadratic terms
                PSk_Py = P @ (S_k @ Py)
                quad1 = self.y @ (P @ (S_j @ PSk_Py))
                
                # Hessian element w.r.t. λⱼ, λₖ
                h_jk = 0.5 * (trace1 + quad1)
                
                # Chain rule
                h_jk *= lambda_vec[j] * lambda_vec[k]
                
                H[j, k] = h_jk
                H[k, j] = h_jk

        return H

    def objective_gradient_hessian(
        self, beta: np.ndarray, log_lambda: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute REML, gradient, and Hessian together (efficient).

        Returns:
            (reml_score, gradient, hessian)
        """
        reml = self.objective(beta, log_lambda)
        grad = self.gradient_wrt_log_lambda(beta, log_lambda)
        hess = self.hessian_wrt_log_lambda(beta, log_lambda)
        return reml, grad, hess

    def _construct_combined_penalty(self, lambda_vec: np.ndarray) -> np.ndarray:
        """Construct combined penalty Sλ = Σⱼ λⱼ Sⱼ."""
        S = np.zeros((self.p, self.p))
        for S_j, lambda_j in zip(self.S_list, lambda_vec):
            S += lambda_j * S_j
        return S


def compute_reml(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    S_list: list[np.ndarray],
    lambda_vec: Optional[np.ndarray] = None,
) -> float:
    """Functional API for REML computation.

    Args:
        X: Design matrix.
        y: Response.
        beta: Fitted coefficients.
        S_list: Penalty matrices.
        lambda_vec: Smoothing parameters.

    Returns:
        REML score.
    """
    from pymgcv.distributions.family_base import GaussianFamily
    
    if lambda_vec is None:
        lambda_vec = np.ones(len(S_list))

    family = GaussianFamily()
    reml_obj = REMLObjective(
        X, y, family, S_list,
        smooth_starts=[0],  # Placeholder
        smooth_sizes=[X.shape[1]],  # Placeholder
    )

    return reml_obj.objective(beta, np.log(lambda_vec))
