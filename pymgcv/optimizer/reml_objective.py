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
        gamma: float = 1.0,
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
            gamma: EDF inflation factor (Wood 2011). gamma > 1 multiplies the
                log|S^+| term so that near-zero-signal smooths are pushed more
                strongly toward λ → ∞ (EDF → 0). Only the log|S^+| term in the
                objective and the rank term in the gradient are scaled; the
                Hessian is unaffected.
        """
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.family = family
        self.S_list = [np.asarray(S, dtype=np.float64) for S in S_list]
        self.smooth_starts = smooth_starts
        self.smooth_sizes = smooth_sizes
        self.offset = np.asarray(offset, dtype=np.float64) if offset is not None else np.zeros_like(y)
        self.dispersion = float(dispersion)
        self.gamma = float(gamma)

        self.n, self.p = self.X.shape
        self.n_smooth = len(S_list)

        # Precompute per-penalty eigenvalues (S_j never changes after construction)
        self._S_eigs_list: list[np.ndarray] = [
            np.linalg.eigvalsh(S_j) for S_j in self.S_list
        ]
        self._S_pos_eigs_list: list[np.ndarray] = [
            eigs[eigs > 1e-10] for eigs in self._S_eigs_list
        ]
        self._penalty_rank_list: list[int] = [
            int(len(pos)) for pos in self._S_pos_eigs_list
        ]
        self._log_S_base: list[float] = [
            float(np.sum(np.log(pos))) if len(pos) > 0 else 0.0
            for pos in self._S_pos_eigs_list
        ]
        # Cache for Mp (null space dim of S_combined); keyed by rounded lambda
        self._mp_cache: tuple | None = None

    def objective(self, beta: np.ndarray, log_lambda: np.ndarray) -> float:
        r"""Compute REML score (to be minimised).

        For Gaussian family, uses the profiled REML (Wood 2011) where sigma^2
        is profiled out analytically:

            REML = (n - Mp) * log(RSS_p / (n - Mp))
                   + log|X'X + S_lambda| - log|S_lambda^+|

        For non-Gaussian families, uses the Laplace-approximate REML:

            REML = deviance/phi + beta' S_lambda beta/phi
                   + log|X'WX + S_lambda| - log|S_lambda^+|

        Args:
            beta: Fitted coefficients at current lambda.
            log_lambda: Log of smoothing parameters.

        Returns:
            REML score (scalar, lower is better).
        """
        lambda_vec = np.exp(log_lambda)
        S_combined = self._construct_combined_penalty(lambda_vec)

        # GLM weight matrix at current beta
        eta = self.X @ beta + self.offset
        mu = self.family.linkinv(eta)
        dmu_deta = self.family.dmu_deta(eta)
        var_mu = np.maximum(self.family.variance(mu, self.dispersion), 1e-10)
        w = (dmu_deta**2) / var_mu

        XtWX = self.X.T @ (self.X * w[:, np.newaxis])
        A = XtWX + S_combined + 1e-7 * np.eye(self.p)

        # Log determinant of A = X'WX + S_lambda
        try:
            sign, logdet = np.linalg.slogdet(A)
            logdet_A = logdet if sign > 0 else 1e15
        except Exception:
            logdet_A = 1e15

        # Log determinant of S_lambda^+ (positive eigenvalues only)
        log_S_plus = self._penalty_log_determinant(lambda_vec)

        # Deviance and penalty
        deviance = -2.0 * self.family.loglik(self.y, mu, self.dispersion)
        penalty = float(beta @ S_combined @ beta)

        # Check if Gaussian (profiled REML)
        from pymgcv.distributions.family_base import GaussianFamily
        S_eigs = np.linalg.eigvalsh(S_combined)
        Mp = int(np.sum(S_eigs < 1e-10))
        n_eff = max(self.n - Mp, 1)
        if isinstance(self.family, GaussianFamily):
            rss_p = deviance + penalty  # = ||y - Xb||^2 + b'Sb for Gaussian
            reml = n_eff * np.log(max(rss_p / n_eff, 1e-300)) + logdet_A - self.gamma * log_S_plus
        else:
            # Working-model profiled REML (matches R mgcv).
            # R evaluates the Gaussian REML on the PIRLS pseudodata where the
            # "RSS" is the Pearson chi-squared: Σ (y-μ)² / V(μ).  For
            # non-canonical links this differs from the deviance.
            pearson = float(np.sum((self.y - mu) ** 2 / var_mu))
            D_p = pearson + penalty
            reml = n_eff * np.log(max(D_p / n_eff, 1e-300)) + logdet_A - self.gamma * log_S_plus

        return reml

    def gradient_wrt_log_lambda(
        self, beta: np.ndarray, log_lambda: np.ndarray
    ) -> np.ndarray:
        r"""Compute gradient of REML w.r.t. log(lambda).

        For Gaussian profiled REML:
            dREML/d(rho_j) = lambda_j * (n-Mp) * beta'*S_j*beta / (dev+pen)
                             + lambda_j * trace(A^{-1} S_j) - rank_j

        For general family:
            dREML/d(rho_j) = lambda_j * [trace(A^{-1} S_j) - beta' S_j beta / phi]
                             - rank_j

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
            A = XtWX + S_combined + 1e-8 * np.eye(self.p)
            A_inv = linalg.pinv(A)

            # Per-smooth penalty ranks
            ranks = self._penalty_ranks()

            # Check if Gaussian (profiled REML gradient)
            from pymgcv.distributions.family_base import GaussianFamily
            penalty = float(beta @ S_combined @ beta)
            S_eigs = np.linalg.eigvalsh(S_combined)
            Mp = int(np.sum(S_eigs < 1e-10))
            n_eff = max(self.n - Mp, 1)

            if isinstance(self.family, GaussianFamily):
                deviance = -2.0 * self.family.loglik(self.y, mu, self.dispersion)
                rss_p = deviance + penalty
            else:
                # Working-model RSS = Pearson chi-squared (matches R mgcv)
                pearson = float(np.sum((self.y - mu) ** 2 / var_mu))
                rss_p = pearson + penalty

            grad = np.zeros(self.n_smooth)
            for j, S_j in enumerate(self.S_list):
                trace_term = lambda_vec[j] * np.trace(A_inv @ S_j)
                bSb = float(beta @ S_j @ beta)
                fit_term = lambda_vec[j] * n_eff * bSb / max(rss_p, 1e-300)
                grad[j] = fit_term + trace_term - self.gamma * ranks[j]

            return grad
        except Exception:
            return np.full(self.n_smooth, np.nan)

    def hessian_wrt_log_lambda(
        self, beta: np.ndarray, log_lambda: np.ndarray
    ) -> np.ndarray:
        r"""Compute Hessian of REML w.r.t. log(lambda).

        H[j,k] = d^2 REML / (d rho_j d rho_k)

        Using the expected Hessian approximation (more stable):
            H[j,k] = lambda_j * lambda_k * trace(A^{-1} S_j A^{-1} S_k)

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
            A = XtWX + S_combined + 1e-8 * np.eye(self.p)

            A_inv = linalg.pinv(A)

            H = np.zeros((self.n_smooth, self.n_smooth))
            # Pre-compute A_inv @ S_j for each j
            AinvS = [A_inv @ S_j for S_j in self.S_list]

            for j in range(self.n_smooth):
                for k in range(j, self.n_smooth):
                    h_jk = lambda_vec[j] * lambda_vec[k] * np.trace(AinvS[j] @ AinvS[k])
                    H[j, k] = h_jk
                    H[k, j] = h_jk

            return H
        except Exception:
            return np.full((self.n_smooth, self.n_smooth), np.nan)

    def objective_gradient_hessian(
        self, beta: np.ndarray, log_lambda: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute REML, gradient, and Hessian in one pass (single A factorisation).

        Factorises A = X'WX + S_lambda exactly once via PenalizedSolver and
        reuses it for the objective, gradient, and Hessian — avoiding the
        inconsistent ridge/ridge mismatch in the separate methods.

        Returns:
            (reml_score, gradient, hessian)
        """
        from pymgcv.linalg.penalized_solver import PenalizedSolver
        from pymgcv.distributions.family_base import GaussianFamily

        lambda_vec = np.exp(log_lambda)
        S_combined = self._construct_combined_penalty(lambda_vec)

        # ---- GLM weights at current beta ----
        eta = self.X @ beta + self.offset
        mu = self.family.linkinv(eta)
        dmu_deta = self.family.dmu_deta(eta)
        var_mu = np.maximum(self.family.variance(mu, self.dispersion), 1e-10)
        w = np.clip((dmu_deta ** 2) / var_mu, 1e-12, 1e8)

        XtWX = self.X.T @ (self.X * w[:, np.newaxis])

        # ---- Single factorisation ----
        solver = PenalizedSolver(XtWX, S_combined)
        logdet_A = solver.log_determinant()
        log_S_plus = self._penalty_log_determinant(lambda_vec)

        deviance = -2.0 * self.family.loglik(self.y, mu, self.dispersion)
        penalty = float(beta @ S_combined @ beta)
        ranks = self._penalty_ranks()

        # ---- Null-space dimension (shared by Gaussian and pseudo-Gaussian REML) ----
        lam_key = tuple(np.round(lambda_vec, 12))
        if self._mp_cache is None or self._mp_cache[0] != lam_key:
            S_eigs = np.linalg.eigvalsh(S_combined)
            Mp = int(np.sum(S_eigs < 1e-10))
            self._mp_cache = (lam_key, Mp)
        Mp = self._mp_cache[1]
        n_eff = max(self.n - Mp, 1)

        # ---- Objective ----
        is_gaussian = isinstance(self.family, GaussianFamily)
        if is_gaussian:
            rss_p = deviance + penalty
            reml = n_eff * np.log(max(rss_p / n_eff, 1e-300)) + logdet_A - self.gamma * log_S_plus
        else:
            # Working-model RSS = Pearson chi-squared (matches R mgcv).
            pearson = float(np.sum((self.y - mu) ** 2 / var_mu))
            rss_p = pearson + penalty
            reml = n_eff * np.log(max(rss_p / n_eff, 1e-300)) + logdet_A - self.gamma * log_S_plus

        if not np.isfinite(reml):
            return float('nan'), np.full(self.n_smooth, np.nan), np.full((self.n_smooth, self.n_smooth), np.nan)

        # ---- Gradient and Hessian via A^{-1} S_j ----
        # Pre-compute A^{-1} S_j for each j (reuse solver)
        AinvS = []
        for j, S_j in enumerate(self.S_list):
            AinvSj = solver.solve(S_j)  # shape (p, p) — columns of A^{-1} S_j
            AinvS.append(AinvSj)

        grad = np.zeros(self.n_smooth)
        H = np.zeros((self.n_smooth, self.n_smooth))
        bSb_arr = np.array([float(beta @ self.S_list[j] @ beta) for j in range(self.n_smooth)])

        if is_gaussian:
            # Gaussian profiled REML: fit_terms = n_eff·λ·bSb / rss_p
            rss_p_val = max(rss_p, 1e-300)
            fit_terms = lambda_vec * n_eff * bSb_arr / rss_p_val
        else:
            # Deviance-based profiled REML: same formula as Gaussian
            rss_p_val = max(rss_p, 1e-300)
            fit_terms = lambda_vec * n_eff * bSb_arr / rss_p_val

        for j in range(self.n_smooth):
            trace_term = lambda_vec[j] * float(np.trace(AinvS[j]))
            grad[j] = fit_terms[j] + trace_term - self.gamma * ranks[j]

            for k in range(j, self.n_smooth):
                # logdet Hessian (positive definite)
                h_jk = lambda_vec[j] * lambda_vec[k] * float(np.trace(AinvS[j] @ AinvS[k]))
                # Fit-term Hessian correction (negative semi-definite)
                h_jk -= fit_terms[j] * fit_terms[k] / max(n_eff * rss_p_val, 1e-300)
                H[j, k] = h_jk
                H[k, j] = h_jk

        return reml, grad, H

    def _construct_combined_penalty(self, lambda_vec: np.ndarray) -> np.ndarray:
        """Construct combined penalty Sλ = Σⱼ λⱼ Sⱼ."""
        S = np.zeros((self.p, self.p))
        for S_j, lambda_j in zip(self.S_list, lambda_vec):
            S += lambda_j * S_j
        return S

    def _penalty_log_determinant(self, lambda_vec: np.ndarray) -> float:
        r"""Compute log|S_lambda^+| = sum_j (rank_j * log(lambda_j) + log|S_j^+|).

        Uses precomputed per-penalty eigenvalues (cached in __init__).
        """
        log_det = 0.0
        for j in range(self.n_smooth):
            r = self._penalty_rank_list[j]
            if r > 0:
                log_det += r * np.log(max(lambda_vec[j], 1e-300))
                log_det += self._log_S_base[j]
        return log_det

    def _penalty_ranks(self) -> list[int]:
        """Return precomputed rank (number of positive eigenvalues) for each penalty."""
        return list(self._penalty_rank_list)


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
