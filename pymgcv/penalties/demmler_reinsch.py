"""Demmler-Reinsch orthogonalization for penalty matrices.

Orthogonalizes the design matrix with respect to a penalty matrix,
improving numerical stability and enabling efficient penalized likelihood computation.

The key idea: Given design matrix X and penalty S, compute orthogonal matrix U such that:
    X̃ = X U   (transformed design matrix)
    D = U^T S U   (penalty is diagonalized)

This separates:
    - Null space: unpenalized basis functions
    - Penalized space: functions subject to smoothness constraint

References:
    - Demmler, A. & Reinsch, C. (1975): Oscillation matrices with spline smoothing
    - Wood, S. N. (2017): Generalized Additive Models, Ch. 4

Module exports:
    - DemmlerReinschOrthogonalization: Main class
    - orthogonalize_design_matrix: Functional API
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


class DemmlerReinschOrthogonalization:
    """Perform Demmler-Reinsch orthogonalization.

    Given:
        X: design matrix, shape (n, p)
        S: penalty matrix, shape (p, p)

    Compute orthogonal U such that:
        D = U^T S U   (diagonal or block-diagonal)
        X̃ = X U       (transformed)

    The transformation separates null space (unpenalized) from penalized basis.

    Attributes:
        X: Original design matrix.
        S: Original penalty matrix.
        U: Orthogonal transformation matrix.
        X_tilde: Transformed design matrix X U.
        D: Diagonalized penalty.
        null_space_dim: Dimension of null space (unpenalized basis).
        penalized_dim: Number of penalized basis functions.
    """

    def __init__(
        self,
        X: np.ndarray,
        S: np.ndarray,
        null_space_dim: int | None = None,
    ) -> None:
        """Initialize Demmler-Reinsch orthogonalization.

        Args:
            X: Design matrix, shape (n, p).
            S: Penalty matrix, shape (p, p), typically positive semi-definite.
            null_space_dim: Dimension of null space (unpenalized). If None, auto-detect.

        Raises:
            ValueError: If X and S dimensions don't match.
        """
        X = np.asarray(X, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)

        if X.shape[1] != S.shape[0] or S.shape[0] != S.shape[1]:
            raise ValueError(
                f"X has shape {X.shape}, S has shape {S.shape}. "
                f"X.shape[1] must equal S.shape[0] == S.shape[1]"
            )

        self.X = X
        self.S = S
        self.n_obs, self.p_cols = X.shape

        # Auto-detect null space dimension (columns where penalty is zero)
        if null_space_dim is None:
            null_space_dim = self._detect_null_space_dim()

        self.null_space_dim = null_space_dim
        self.penalized_dim = self.p_cols - null_space_dim

        # Perform orthogonalization
        self.U: np.ndarray = np.eye(self.p_cols)
        self.X_tilde: np.ndarray = X.copy()
        self.D: np.ndarray = S.copy()

        self._orthogonalize()

    def _detect_null_space_dim(self) -> int:
        """Auto-detect dimension of null space (zero eigenvalues of S).

        Returns:
            Number of zero eigenvalues, i.e., unpenalized dimensions.
        """
        eigenvalues = np.linalg.eigvalsh(self.S)
        # Count eigenvalues < 1e-10
        null_dim = np.sum(eigenvalues < 1e-10)
        return max(0, null_dim)

    def _orthogonalize(self) -> None:
        """Perform the orthogonalization via eigendecomposition.

        Steps:
        1. Compute eigendecomposition: S = V D V^T
        2. Order eigenvectors by eigenvalue (zero first)
        3. Set U = V
        4. Transform: X̃ = X U, D̃ = U^T S U
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(self.S)

        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # U is the eigenvector matrix
        self.U = eigenvectors

        # Transform design matrix
        self.X_tilde = self.X @ self.U

        # Diagonalized penalty (eigenvalues on diagonal)
        self.D = np.diag(eigenvalues)

    def transformed_design_matrix(self) -> np.ndarray:
        """Return transformed design matrix X̃ = X U.

        Shape: (n, p)
        """
        return self.X_tilde

    def diagonalized_penalty(self) -> np.ndarray:
        """Return diagonalized penalty matrix D = U^T S U.

        Shape: (p, p), diagonal.
        """
        return self.D

    def transformation_matrix(self) -> np.ndarray:
        """Return transformation matrix U.

        Shape: (p, p)
        """
        return self.U

    def inverse_transform(self, beta_tilde: np.ndarray) -> np.ndarray:
        """Transform coefficients back to original basis.

        Given β̃ in the transformed basis, compute β = U β̃.

        Args:
            beta_tilde: Coefficients in transformed basis.

        Returns:
            Coefficients in original basis.
        """
        return self.U @ beta_tilde

    def decompose_into_null_penalized(self, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decompose coefficients into null space and penalized parts.

        Returns:
            (beta_null, beta_penalized) where:
            - beta_null: first null_space_dim coefficients
            - beta_penalized: remaining coefficients (subject to penalty)
        """
        beta_tilde = self.U.T @ beta
        beta_null = beta_tilde[: self.null_space_dim]
        beta_penalized = beta_tilde[self.null_space_dim :]
        return beta_null, beta_penalized

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "Demmler-Reinsch Orthogonalization",
            "==================================",
            f"Design matrix shape: {self.X.shape}",
            f"Penalty matrix shape: {self.S.shape}",
            f"Null space dimension: {self.null_space_dim}",
            f"Penalized dimension: {self.penalized_dim}",
            f"Transformation matrix U: {self.U.shape}",
            "Diagonalized penalty (eigenvalues):",
        ]
        eigenvalues = np.diag(self.D)
        for i, ev in enumerate(eigenvalues):
            lines.append(f"  λ[{i}] = {ev:.6e}")
        return "\n".join(lines)


def orthogonalize_design_matrix(
    X: np.ndarray,
    S: np.ndarray,
    null_space_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Functional API for Demmler-Reinsch orthogonalization.

    Args:
        X: Design matrix, shape (n, p).
        S: Penalty matrix, shape (p, p).
        null_space_dim: Dimension of null space. If None, auto-detect.

    Returns:
        (X_tilde, D, U) where:
        - X_tilde: Transformed design matrix
        - D: Diagonalized penalty
        - U: Transformation matrix
    """
    dr = DemmlerReinschOrthogonalization(X, S, null_space_dim)
    return dr.transformed_design_matrix(), dr.diagonalized_penalty(), dr.transformation_matrix()
