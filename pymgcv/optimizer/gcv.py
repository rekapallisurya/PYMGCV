"""Generalized Cross-Validation (GCV) for smoothing parameter selection.

GCV is the most common criterion for automatic smoothing parameter selection in GAMs.
It balances model fit and complexity without requiring knowledge of σ².

Theory:
    GCV(λ) = n·D(λ) / (n - DoF(λ))²
    
    where:
    - D(λ) = deviance (sum of squared residuals for Gaussian)
    - DoF(λ) = trace(A) = effective degrees of freedom
    - A = X(XᵀX + Σλⱼ Sⱼ)⁻¹Xᵀ = hat matrix
    - n = sample size
    - λⱼ = smoothing parameters

The GCV criterion avoids expensive cross-validation by using a closed-form approximation.

References:
    - Craven, P. and Wahba, G. (1978). Smoothing noisy data with spline functions.
      Numerische Mathematik, 31, 377-403.
    - Wood, S.N. (2004). Stable and efficient multiple smoothing parameter
      estimation for generalized additive models. JASA, 99(467), 673-686.

Module exports:
    - GCVCriterion: Main GCV objective class
    - compute_gcv: Function to compute GCV score
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg


class GCVCriterion:
    """Generalized Cross-Validation criterion for GAM smoothing parameters.

    Computes GCV(λ) = n·D(λ) / (n - DoF(λ))² where DoF = trace(A).

    This is the most commonly used criterion for automatic smoothing parameter
    selection in GAMs. It is computationally efficient and works without
    requiring knowledge of σ².

    Attributes:
        X: Design matrix, shape (n, p).
        y: Response vector, shape (n,).
        S_list: List of penalty matrices, one per smooth term.
        lambda_vec: Current smoothing parameters λ.
        offset: Offset vector, shape (n,).
        n: Sample size.
        tol: Tolerance for numerical stability.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        S_list: list[np.ndarray],
        lambda_vec: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        tol: float = 1e-10,
    ) -> None:
        """Initialize GCV criterion.

        Args:
            X: Design matrix, shape (n, p).
            y: Response vector, shape (n,).
            S_list: List of penalty matrices for each smooth term.
            lambda_vec: Initial smoothing parameters. If None, defaults to ones.
            offset: Offset vector. If None, defaults to zeros.
            tol: Tolerance for numerical stability (for trace computation).

        Raises:
            ValueError: If shapes are inconsistent.
        """
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.n, self.p = self.X.shape

        if len(self.y) != self.n:
            raise ValueError(f'y length {len(self.y)} != X rows {self.n}')

        self.S_list = [np.asarray(S, dtype=np.float64) for S in S_list]
        
        # Validate penalty matrix dimensions
        for i, S in enumerate(self.S_list):
            if S.shape != (self.p, self.p):
                raise ValueError(
                    f'Penalty matrix {i} has shape {S.shape}, '
                    f'expected ({self.p}, {self.p})'
                )

        # Initialize smoothing parameters
        if lambda_vec is None:
            self.lambda_vec = np.ones(len(S_list))
        else:
            self.lambda_vec = np.asarray(lambda_vec, dtype=np.float64)

        if len(self.lambda_vec) != len(self.S_list):
            raise ValueError(
                f'lambda_vec length {len(self.lambda_vec)} '
                f'!= # penalties {len(self.S_list)}'
            )

        # Offset vector
        self.offset = (
            np.asarray(offset, dtype=np.float64)
            if offset is not None
            else np.zeros(self.n)
        )
        
        if len(self.offset) != self.n:
            raise ValueError(f'offset length {len(self.offset)} != X rows {self.n}')

        self.tol = float(tol)
        self.n_smooth = len(S_list)

        # Cache for efficiency
        self._XTX: Optional[np.ndarray] = None
        self._S_combined: Optional[np.ndarray] = None
        self._A_trace: Optional[float] = None

    def set_lambda(self, lambda_vec: np.ndarray) -> None:
        """Update smoothing parameters and clear cache.

        Args:
            lambda_vec: New smoothing parameters.
        """
        self.lambda_vec = np.asarray(lambda_vec, dtype=np.float64)
        self._S_combined = None
        self._A_trace = None

    def _construct_combined_penalty(self) -> np.ndarray:
        """Construct combined penalty: Sλ = Σⱼ λⱼ Sⱼ."""
        if self._S_combined is not None:
            return self._S_combined

        S = np.zeros((self.p, self.p))
        for S_j, lambda_j in zip(self.S_list, self.lambda_vec):
            S += lambda_j * S_j

        self._S_combined = S
        return S

    def _compute_XTX(self) -> np.ndarray:
        """Compute X'X (with caching)."""
        if self._XTX is None:
            self._XTX = self.X.T @ self.X
        return self._XTX

    def _compute_trace_hat_matrix(self, beta: np.ndarray) -> float:
        """Compute trace of hat matrix A = X(X'X + Sλ)⁻¹X'.

        This is done efficiently without forming A explicitly.
        DoF = trace(A) = trace(X(X'X + Sλ)⁻¹X')

        Args:
            beta: Coefficient vector (unused, kept for API consistency).

        Returns:
            Effective degrees of freedom (trace of hat matrix).
        """
        try:
            XTX = self._compute_XTX()
            S_lambda = self._construct_combined_penalty()
            
            # Form system matrix
            H = XTX + S_lambda
            
            # Compute trace: trace(X(X'X + Sλ)⁻¹X')
            # = trace((X'X + Sλ)⁻¹ X'X)
            H_inv = linalg.inv(H)
            trace_dof = np.trace(H_inv @ XTX)
            
            return float(np.clip(trace_dof, 0, self.p))
        except linalg.LinAlgError as e:
            # Return fallback if matrix is singular
            return float(min(self.n, self.p))

    def deviance(self, beta: np.ndarray) -> float:
        """Compute deviance (sum of squared residuals).

        For Gaussian likelihood:
            D = Σ(yᵢ - ŷᵢ)²

        Args:
            beta: Coefficient vector.

        Returns:
            Deviance value.
        """
        eta = self.X @ beta + self.offset
        residuals = self.y - eta
        dev = np.sum(residuals ** 2)
        return float(dev)

    def gcv(self, beta: Optional[np.ndarray] = None) -> float:
        """Compute GCV score.

        GCV(λ) = n·D(λ) / (n - DoF(λ))²

        Args:
            beta: Coefficient vector fitted for current λ.
                 If None, beta=0 is assumed (for initialization).

        Returns:
            GCV score (lower is better).
        """
        if beta is None:
            beta = np.zeros(self.p)

        dev = self.deviance(beta)

        # Effective degrees of freedom
        dof = self._compute_trace_hat_matrix(beta)

        # Avoid division by zero
        denom_dof = max(self.n - dof, 0.01)

        # GCV = n * D / (n - DoF)²
        gcv_score = self.n * dev / (denom_dof ** 2)

        return float(gcv_score)

    def compute_residual_variance(self, beta: np.ndarray) -> float:
        """Compute estimate of residual variance.

        σ̂² = D / (n - DoF)

        This can be used as an estimate of the scale parameter,
        or compared with assumed values.

        Args:
            beta: Coefficient vector.

        Returns:
            Estimated residual variance.
        """
        dev = self.deviance(beta)
        dof = self._compute_trace_hat_matrix(beta)
        denom = max(self.n - dof, 1)
        return float(dev / denom)

    def __call__(self, beta: np.ndarray) -> float:
        """Make criterion callable for optimization.

        Returns negative GCV to be minimized by scipy optimizers.

        Args:
            beta: Coefficient vector.

        Returns:
            Negative GCV score (for minimization).
        """
        return -self.gcv(beta)


def compute_gcv(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    S_list: list[np.ndarray],
    lambda_vec: np.ndarray,
    offset: Optional[np.ndarray] = None,
) -> float:
    """Compute GCV score for given β and λ.

    Functional API for GCV computation.

    Args:
        X: Design matrix.
        y: Response vector.
        beta: Coefficient vector.
        S_list: List of penalty matrices.
        lambda_vec: Smoothing parameters.
        offset: Offset vector.

    Returns:
        GCV score.
    """
    criterion = GCVCriterion(X, y, S_list, lambda_vec, offset)
    return criterion.gcv(beta)


def compare_gcv_scores(
    scores: dict[str, float],
) -> tuple[str, float]:
    """Find best model by GCV score.

    Args:
        scores: Dictionary mapping model name to GCV score.

    Returns:
        (best_model_name, best_gcv_score)
    """
    if not scores:
        raise ValueError('No scores provided')
    
    best_name = min(scores, key=lambda k: scores[k])
    best_score = scores[best_name]
    return best_name, best_score
