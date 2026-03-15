"""Penalty matrix construction for smooth terms.

Constructs penalty matrices S_j for each smooth term.
The penalty for smooth term j is: βⱼᵀ Sⱼ βⱼ

For thin plate regression splines, the penalty is derived from the
roughness of the function (second derivative integral).

References:
    - Green & Silverman (1994): Nonparametric Regression & Generalized Linear Models
    - Wood (2003): Thin plate regression splines
    - Wand & Ormerod (2008): On semiparametric regression with doubly robust estimation

Module exports:
    - PenaltyMatrix: Main penalty matrix constructor
    - construct_penalty_matrix: Functional API
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg, sparse


class PenaltyMatrix:
    """Construct penalty matrices for smooth terms.

    For a smooth basis with k basis functions, constructs a (k × k) penalty matrix S
    that penalizes roughness/complexity.

    The penalty contribution to the objective is: βᵀ S β

    Different smooth types have different penalty constructions:
    - TPRS: Penalizes curvature
    - Cubic spline: Penalizes second derivative
    - Random effect: Identity matrix (equivalent to variance prior)

    Attributes:
        basis_dim: Dimension of the basis (k).
        penalty_type: Type of penalty ('tprs', 'spline', 'random-effect').
        S: Penalty matrix, shape (k, k), typically sparse.
        eigenvalues: Eigenvalues of S (for diagnostics).
        eigenvectors: Eigenvectors of S.
    """

    def __init__(
        self,
        basis_dim: int,
        penalty_type: str = 'tprs',
        data: Optional[np.ndarray] = None,
        knots: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize penalty matrix constructor.

        Args:
            basis_dim: Dimension of the basis (k).
            penalty_type: One of 'tprs', 'cubic-spline', 'random-effect'.
            data: Data used for basis construction (for autocovariance, etc.).
            knots: Knot locations (for spline-based penalties).

        Raises:
            ValueError: If penalty_type is unknown.
        """
        self.basis_dim = basis_dim
        self.penalty_type = penalty_type.lower()
        self.data = data
        self.knots = knots

        self.S: np.ndarray = np.zeros((basis_dim, basis_dim))
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None

        if self.penalty_type == 'tprs':
            self._construct_tprs_penalty()
        elif self.penalty_type == 'cubic-spline':
            self._construct_spline_penalty()
        elif self.penalty_type == 'random-effect':
            self._construct_random_effect_penalty()
        else:
            raise ValueError(f'Unknown penalty type: {penalty_type}')

    def _construct_tprs_penalty(self) -> None:
        """Construct penalty for thin plate regression splines.

        For TPRS with univariate input, the penalty matrix comes from the
        thin plate spline roughness penalty:

        penalty = ∫∫ (∂²f/∂x²)² dx

        In the basis space, this is approximated via the RBF distance matrix
        and eigendecomposition.

        For now, use a simple approximation: second-order differences.
        """
        # Simple second-difference penalty
        # This penalizes second derivative: β² - 2β_i β_{i+1} + β_{i+1}²
        k = self.basis_dim

        # Construct second-difference matrix D
        D = np.eye(k) - 2 * np.eye(k, k, k=1) + np.eye(k, k, k=2)
        
        # Remove last two rows (incomplete differences)
        if k > 2:
            D = D[:-2]
        
        # Penalty matrix: S = D^T D
        self.S = D.T @ D

        # Compute eigendecomposition for diagnostics
        self._compute_eigendecomposition()

    def _construct_spline_penalty(self) -> None:
        """Construct penalty for cubic splines.

        For cubic splines, the penalty is based on the second derivative:
        penalty = ∫ (f'')² dx

        In the knot basis, this is computed via finite differences.
        """
        # Similar to TPRS but potentially with refinements for knot spacing
        self._construct_tprs_penalty()  # TODO: Refine for spline-specific structure

    def _construct_random_effect_penalty(self) -> None:
        """Construct penalty for random effects (random intercept per group).

        For random effects: s(group, bs='re')
        The penalty is essentially identity: β ~ N(0, σ²I)
        Which corresponds to penalty matrix S = I (up to scaling).
        """
        self.S = np.eye(self.basis_dim)
        self._compute_eigendecomposition()

    def _compute_eigendecomposition(self) -> None:
        """Compute eigendecomposition of S for diagnostics and REML."""
        try:
            eigenvalues, eigenvectors = linalg.eigh(self.S)
            # Sort by ascending eigenvalue
            idx = np.argsort(eigenvalues)
            self.eigenvalues = eigenvalues[idx]
            self.eigenvectors = eigenvectors[:, idx]
        except linalg.LinAlgError:
            # If S is singular or near-singular, use SVD
            u, svals, vt = linalg.svd(self.S)
            self.eigenvalues = svals
            self.eigenvectors = u

    def penalty_matrix(self) -> np.ndarray:
        """Return the penalty matrix S, shape (k, k)."""
        return self.S

    def penalty_matrix_sparse(self) -> sparse.csr_matrix:
        """Return penalty matrix as sparse CSR format (for efficiency)."""
        return sparse.csr_matrix(self.S)

    def compute_edf_penalty(self, H: np.ndarray, lambda_val: float) -> float:
        """Compute contributing EDF from this penalty to total EDF.

        EDF contribution = trace(H(:, j) * H(j, :)) where j are columns
        corresponding to this smooth.

        For now, return a placeholder.

        Args:
            H: Penalized hat matrix.
            lambda_val: Smoothing parameter λ.

        Returns:
            Effective degrees of freedom from this smooth.
        """
        # TODO: Implement proper EDF contribution
        return float(self.basis_dim)


class PenaltyMatrixSet:
    """Manage multiple penalty matrices (one per smooth term).

    Combines: Sλ = Σⱼ λⱼ Sⱼ

    Attributes:
        penalties: List of PenaltyMatrix objects.
        lambda_vals: Smoothing parameters λⱼ (default: 1.0).
    """

    def __init__(self, penalties: list[PenaltyMatrix]) -> None:
        """Initialize set of penalty matrices.

        Args:
            penalties: List of PenaltyMatrix objects (one per smooth term).
        """
        self.penalties = penalties
        self.n_penalties = len(penalties)
        self.lambda_vals = np.ones(self.n_penalties)

    def set_lambda(self, lambda_vals: np.ndarray) -> None:
        """Set smoothing parameters λ.

        Args:
            lambda_vals: Array of smoothing parameters, length n_penalties.

        Raises:
            ValueError: If length doesn't match number of penalties.
        """
        if len(lambda_vals) != self.n_penalties:
            raise ValueError(
                f'Expected {self.n_penalties} smoothing parameters, '
                f'got {len(lambda_vals)}'
            )
        self.lambda_vals = np.asarray(lambda_vals, dtype=float)

    def combined_penalty(self, indices: list[slice]) -> np.ndarray:
        """Construct combined penalty matrix Σⱼ λⱼ Sⱼ.

        Args:
            indices: List of slices indicating which columns belong to each smooth.

        Returns:
            Combined penalty matrix, shape (p, p) where p = total design matrix cols.
        """
        n_cols = sum(idx.stop - idx.start for idx in indices)
        S_combined = np.zeros((n_cols, n_cols))

        for j, (penalty, idx) in enumerate(zip(self.penalties, indices)):
            S_j = penalty.penalty_matrix()
            lambda_j = self.lambda_vals[j]
            
            # Place λⱼ Sⱼ in the appropriate block
            i_start, i_stop = idx.start, idx.stop
            S_combined[i_start:i_stop, i_start:i_stop] += lambda_j * S_j

        return S_combined

    def gradient_wrt_lambda(
        self,
        H: np.ndarray,
        indices: list[slice],
        lambda_j: int,
    ) -> float:
        """Compute gradient of objective w.r.t. λⱼ.

        Used in MAGIC optimizer to update smoothing parameters.

        ∂REML / ∂λⱼ ∝ trace((H Sⱼ)²)

        Args:
            H: Penalized hat matrix, shape (n, n).
            indices: Indices for smooth terms.
            lambda_j: Index of smoothing parameter to compute gradient for.

        Returns:
            Gradient w.r.t. λⱼ.
        """
        idx = indices[lambda_j]
        S_j = self.penalties[lambda_j].penalty_matrix()

        # Mark columns of interest in H
        H_j = H[idx, idx]

        # Compute trace(H_j Sⱼ)² ≈ sum((H_j Sⱼ)²)
        HS = H_j @ S_j
        grad = np.sum(HS**2)

        return float(grad)


def construct_penalty_matrix(
    basis_dim: int,
    penalty_type: str = 'tprs',
) -> np.ndarray:
    """Functional API for penalty matrix construction.

    Args:
        basis_dim: Dimension of the basis.
        penalty_type: Type of penalty ('tprs', 'cubic-spline', 'random-effect').

    Returns:
        Penalty matrix S, shape (basis_dim, basis_dim).
    """
    pm = PenaltyMatrix(basis_dim, penalty_type)
    return pm.penalty_matrix()
