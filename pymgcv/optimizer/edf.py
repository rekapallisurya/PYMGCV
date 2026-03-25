"""Effective degrees of freedom (EDF) computation.

EDF measures the complexity of a fitted GAM model. For smooth terms, it quantifies
how many "parameters" are being used after penalization.

Key quantities:
    - Total EDF = trace(H) where H is the penalized hat matrix.
    - Smooth-specific EDF = trace(H[smooth_indices, :])
    - Ref.df = number of basis functions before penalization

References:
    - Wood, S. N. (2017): Generalized Additive Models, Ch. 4-5
    - Hastie, T. & Tibshirani, R. (1990): Generalized Additive Models

Module exports:
    - EDFComputer: Main class
    - compute_edf: Functional API
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


class EDFComputer:
    """Compute effective degrees of freedom (EDF) for GAM models.

    Attributes:
        X: Design matrix, shape (n, p).
        S_combined: Combined penalty matrix, shape (p, p).
        H: Penalized hat matrix, shape (n, n).
        edf_total: Total EDF.
        edf_smooth: EDF per smooth term.
        ref_df: Reference (unpenalized) degrees of freedom.
    """

    def __init__(
        self,
        X: np.ndarray,
        S_combined: np.ndarray,
        family: object,
        beta: np.ndarray,
        offset: np.ndarray | None = None,
        dispersion: float = 1.0,
    ) -> None:
        """Initialize EDF computer.

        Args:
            X: Design matrix, shape (n, p).
            S_combined: Combined penalty matrix (Gaussian case).
            family: Distribution family (for GLM weight matrix).
            beta: Fitted coefficients.
            offset: Offset vector.
            dispersion: Dispersion parameter.
        """
        self.X = np.asarray(X, dtype=np.float64)
        self.S_combined = np.asarray(S_combined, dtype=np.float64)
        self.family = family
        self.beta = np.asarray(beta, dtype=np.float64)
        self.offset = (
            np.asarray(offset, dtype=np.float64)
            if offset is not None
            else np.zeros(self.X.shape[0])
        )
        self.dispersion = float(dispersion)

        self.n, self.p = self.X.shape

        # Compute influence matrix F (p x p)
        self.F: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self._compute_hat_matrix()

        # EDF components
        self.edf_total = 0.0
        self.edf_smooth: dict[int, float] = {}
        self.ref_df = self.p

        self._compute_edf()

    def _compute_hat_matrix(self) -> None:
        """Compute the p x p influence matrix F = A^{-1} X^T W X.

        EDF = trace(F), which equals trace of the n x n hat matrix H
        but is computed in O(np^2) instead of O(n^3).

        For per-smooth EDF, F[j,j] diagonal entries corresponding to
        smooth j's columns give the EDF contribution.
        """
        eta = self.X @ self.beta + self.offset
        mu = self.family.linkinv(eta)
        dmu_deta = self.family.dmu_deta(eta)
        var_mu = np.maximum(self.family.variance(mu, self.dispersion), 1e-10)

        w = (dmu_deta**2) / var_mu

        XtWX = self.X.T @ (self.X * w[:, np.newaxis])

        A = XtWX + self.S_combined

        # Use Cholesky for numerical stability, fall back to lstsq
        try:
            L = linalg.cholesky(A, lower=True)
            A_inv_XtWX = linalg.cho_solve((L, True), XtWX)
            self.F = A_inv_XtWX
        except linalg.LinAlgError:
            A_inv_XtWX = linalg.lstsq(A, XtWX)[0]
            self.F = A_inv_XtWX
        # Store as H for backward compatibility (now p x p, not n x n)
        self.H = self.F

    def _compute_edf(self) -> None:
        """Compute total and smooth-term EDF.

        Total EDF = trace(H)

        For each smooth term (with columns in range [start, stop]):
            EDF_smooth = sum(H[i, i]) where i in [start, stop]
        """
        if self.H is None:
            return

        # Total EDF
        self.edf_total = np.trace(self.H)

    def edf_by_smooth(self, smooth_indices: list[slice]) -> dict[int, float]:
        """Compute EDF for each smooth term.

        Args:
            smooth_indices: List of slices indicating column ranges for each smooth.

        Returns:
            Dict mapping smooth index -> EDF for that smooth.
        """
        if self.F is None:
            return {}

        edf_dict = {}
        for j, idx in enumerate(smooth_indices):
            # Sum of diagonal elements of F for this smooth's columns
            edf_smooth = np.sum(np.diag(self.F)[idx])
            edf_dict[j] = float(edf_smooth)

        return edf_dict

    def total_edf(self) -> float:
        """Return total EDF."""
        return self.edf_total

    def summary(self) -> str:
        """Summary of EDF."""
        lines = [
            "Effective Degrees of Freedom",
            "=============================",
            f"Total EDF: {self.edf_total:.4f}",
            f"Reference DF: {self.ref_df}",
            f"Ratio (EDF/Ref): {self.edf_total / self.ref_df:.4f}",
        ]
        return "\n".join(lines)


def compute_edf(
    X: np.ndarray,
    S_combined: np.ndarray,
    family: object,
    beta: np.ndarray,
    offset: np.ndarray | None = None,
    dispersion: float = 1.0,
) -> float:
    """Compute total effective degrees of freedom.

    Args:
        X: Design matrix.
        S_combined: Combined penalty matrix.
        family: Distribution family.
        beta: Fitted coefficients.
        offset: Offset vector.
        dispersion: Dispersion parameter.

    Returns:
        Total EDF (scalar).
    """
    computer = EDFComputer(X, S_combined, family, beta, offset, dispersion)
    return computer.total_edf()


def compute_edf_per_smooth(
    X: np.ndarray,
    S_combined: np.ndarray,
    family: object,
    beta: np.ndarray,
    smooth_indices: list[slice],
    offset: np.ndarray | None = None,
    dispersion: float = 1.0,
) -> dict[int, float]:
    """Compute EDF per smooth term.

    Args:
        X: Design matrix.
        S_combined: Combined penalty matrix.
        family: Distribution family.
        beta: Fitted coefficients.
        smooth_indices: Column slices for each smooth term.
        offset: Offset vector.
        dispersion: Dispersion parameter.

    Returns:
        Dict: smooth_index → EDF
    """
    computer = EDFComputer(X, S_combined, family, beta, offset, dispersion)
    return computer.edf_by_smooth(smooth_indices)
