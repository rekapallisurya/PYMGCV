"""Penalized likelihood objective for GAM fitting.

Formulates the penalized likelihood objective:

    L(β) = -2 log L(y | β) + βᵀ Σⱼ λⱼ Sⱼ β

where:
    - -2 log L is the deviance for the chosen family
    - λⱼ are smoothing parameters
    - Sⱼ are penalty matrices (one per smooth term)

For Gaussian family: L(β) = Σ(y - Xβ)² / σ² + βᵀ S β
For Poisson: L(β) = -2 Σ(y log μ - μ) + βᵀ S β    where μ = exp(Xβ)

Module exports:
    - PenalizedLikelihood: Main objective class
    - GaussianPenalizedLikelihood: Specialized for Gaussian family
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PenalizedLikelihood:
    """Penalized likelihood objective for GAM fitting.

    Encapsulates:
        - Design matrix X and response y
        - Family distribution (Gaussian, Poisson, Gamma, Tweedie)
        - Penalty matrices and smoothing parameters
        - Computation of objective, gradient, Hessian

    Attributes:
        X: Design matrix, shape (n, p).
        y: Response vector, shape (n,).
        offset: Offset vector (for Poisson/Tweedie), shape (n,).
        family: Distribution family object.
        S_list: List of penalty matrices, one per smooth term.
        lambda_vec: Smoothing parameters λⱼ.
        current_beta: Current coefficient estimate.
        dispersion: Dispersion parameter φ (for Gamma, Tweedie).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        family: object,
        S_list: list[np.ndarray],
        lambda_vec: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        dispersion: float = 1.0,
    ) -> None:
        """Initialize penalized likelihood.

        Args:
            X: Design matrix, shape (n, p).
            y: Response vector, shape (n,).
            family: Family object (Gaussian, Poisson, etc.) with variance(), linkinv(), etc.
            S_list: List of penalty matrices, shape (p, p) each. One per smooth term.
            lambda_vec: Smoothing parameters. If None, defaults to ones.
            offset: Offset vector for Poisson/Tweedie models.
            dispersion: Dispersion parameter φ (default 1.0).
        """
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.family = family
        self.S_list = [np.asarray(S, dtype=np.float64) for S in S_list]
        self.offset = np.asarray(offset, dtype=np.float64) if offset is not None else np.zeros_like(y)
        self.dispersion = float(dispersion)

        self.n, self.p = self.X.shape

        # Default smoothing parameters
        if lambda_vec is None:
            lambda_vec = np.ones(len(S_list))
        self.lambda_vec = np.asarray(lambda_vec, dtype=np.float64)

        if len(self.lambda_vec) != len(self.S_list):
            raise ValueError(
                f'Number of smoothing parameters ({len(self.lambda_vec)}) '
                f'does not match number of penalty matrices ({len(self.S_list)})'
            )

        # Construct combined penalty matrix
        self.S = self._construct_combined_penalty()

        # Current coefficient estimate
        self.current_beta = np.zeros(self.p)
        self.current_eta: Optional[np.ndarray] = None  # Linear predictor
        self.current_mu: Optional[np.ndarray] = None  # Mean

    def _construct_combined_penalty(self) -> np.ndarray:
        """Construct combined penalty Sλ = Σⱼ λⱼ Sⱼ."""
        S = np.zeros((self.p, self.p))
        for S_j, lambda_j in zip(self.S_list, self.lambda_vec):
            S += lambda_j * S_j
        return S

    def set_lambda(self, lambda_vec: np.ndarray) -> None:
        """Update smoothing parameters and recompute penalty."""
        if len(lambda_vec) != len(self.S_list):
            raise ValueError(f'Expected {len(self.S_list)} smoothing parameters')
        self.lambda_vec = np.asarray(lambda_vec, dtype=np.float64)
        self.S = self._construct_combined_penalty()

    def objective(self, beta: np.ndarray) -> float:
        r"""Compute penalized likelihood objective.

        L(β) = deviance(β) + βᵀ Sλ β

        Args:
            beta: Coefficient vector, shape (p,).

        Returns:
            Scalar objective value.
        """
        # Linear predictor
        eta = self.X @ beta + self.offset
        
        # Mean (via link function)
        mu = self.family.linkinv(eta)

        # Deviance
        deviance = self._compute_deviance(mu)

        # Penalty
        penalty = beta @ self.S @ beta

        return deviance + penalty

    def objective_and_gradient(self, beta: np.ndarray) -> tuple[float, np.ndarray]:
        r"""Compute objective and gradient.

        ∇L = Xᵀ W (y - μ) + 2 Sλ β

        Args:
            beta: Coefficient vector.

        Returns:
            (objective, gradient)
        """
        obj = self.objective(beta)

        # Linear predictor and mean
        eta = self.X @ beta + self.offset
        mu = self.family.linkinv(eta)

        # Gradient computation
        # For GLMs: ∇L = -Xᵀ (y - μ) * (dμ/dη) + 2 Sλ β
        # Variance function
        var_mu = self.family.variance(mu, self.dispersion)
        
        # Gradient of deviance
        grad_deviance = -self.X.T @ ((self.y - mu) / var_mu * self.family.dmu_deta(eta))
        
        # Penalty gradient
        grad_penalty = 2 * self.S @ beta

        grad = grad_deviance + grad_penalty

        return obj, grad

    def _compute_deviance(self, mu: np.ndarray) -> float:
        """Compute deviance for the family.

        Args:
            mu: Predicted mean vector.

        Returns:
            Deviance (scalar).
        """
        return -2 * self.family.loglik(self.y, mu, self.dispersion)

    def hessian(self, beta: np.ndarray) -> np.ndarray:
        r"""Compute Hessian of penalized likelihood.

        H = Xᵀ W X + 2 Sλ

        where W is the GLM weight matrix (diagonal).

        Args:
            beta: Coefficient vector.

        Returns:
            Hessian matrix, shape (p, p).
        """
        eta = self.X @ beta + self.offset
        mu = self.family.linkinv(eta)
        
        # GLM weight matrix
        var_mu = self.family.variance(mu, self.dispersion)
        dmu_deta = self.family.dmu_deta(eta)
        
        # W is diagonal: w_ii = (dμ/dη)² / Var(y)
        w = (dmu_deta**2) / var_mu
        
        # X^T W X
        XtWX = self.X.T @ (self.X * w[:, np.newaxis])
        
        # Penalty Hessian
        H_penalty = 2 * self.S
        
        return XtWX + H_penalty

    def summary(self) -> str:
        """Return summary of the objective."""
        lines = [
            'Penalized Likelihood',
            '====================',
            f'Observations: {self.n}',
            f'Coefficients: {self.p}',
            f'Family: {self.family.__class__.__name__}',
            f'Dispersion: {self.dispersion:.6f}',
            f'Smoothing parameters: {self.lambda_vec}',
            f'Combined penalty rank: {np.linalg.matrix_rank(self.S)}',
        ]
        return '\n'.join(lines)


class GaussianPenalizedLikelihood(PenalizedLikelihood):
    """Specialized penalized likelihood for Gaussian family.

    For Gaussian: deviance = Σ(y - Xβ)²

    Objective: L(β) = ||y - Xβ||² + βᵀ Sλ β

    This has a closed-form solution (not iterative).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        S_list: list[np.ndarray],
        lambda_vec: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize Gaussian penalized likelihood.

        Args:
            X: Design matrix.
            y: Response vector.
            S_list: Penalty matrices.
            lambda_vec: Smoothing parameters.
        """
        # Create a dummy Gaussian family
        from pymgcv.distributions.family_base import GaussianFamily
        family = GaussianFamily()
        
        super().__init__(
            X=X,
            y=y,
            family=family,
            S_list=S_list,
            lambda_vec=lambda_vec,
            offset=None,
            dispersion=1.0,
        )

    def objective(self, beta: np.ndarray) -> float:
        r"""Compute objective for Gaussian.

        L(β) = ||y - Xβ||² + βᵀ Sλ β
        """
        residuals = self.y - self.X @ beta
        ssr = np.sum(residuals**2)
        penalty = beta @ self.S @ beta
        return ssr + penalty

    def solve_closed_form(self) -> np.ndarray:
        r"""Solve Gaussian penalized likelihood in closed form.

        (Xᵀ X + Sλ) β = Xᵀ y

        Returns:
            Solution β.
        """
        from scipy import linalg
        
        XtX = self.X.T @ self.X
        Xty = self.X.T @ self.y
        
        # Solve (X^T X + S) β = X^T y
        A = XtX + self.S
        beta = linalg.solve(A, Xty)
        
        return beta
