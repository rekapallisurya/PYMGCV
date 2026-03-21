"""Linear algebra utilities for numerical stability.

Provides efficient implementations of:
- Cholesky decomposition and solving
- QR decomposition and solving
- Eigendecomposition
- Trace computations via conjugate gradient
- Efficient matrix inversion
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .native_engine import solve_spd_fortran


def solve_symmetric_definite(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve symmetric positive definite system Ax = b.

    Args:
        A: SPD matrix.
        b: Right-hand side vector.

    Returns:
        Solution x.
    """
    return solve_spd_fortran(A, b)


def trace_matrix_inverse_product(A: np.ndarray, B: np.ndarray) -> float:
    """Compute trace(A^{-1} B) efficiently via Cholesky factorisation.

    Solves A Y = B using the Cholesky factor so that Y = A^{-1} B,
    then returns trace(Y).  Avoids forming A^{-1} explicitly.
    Falls back to explicit inversion when A is not positive definite.

    Args:
        A: SPD matrix of shape (p, p).
        B: Matrix of shape (p, p).

    Returns:
        trace(A^{-1} B) as a float.
    """
    try:
        cho = linalg.cho_factor(A, lower=True, check_finite=False)
        Y = linalg.cho_solve(cho, B, check_finite=False)
        return float(np.trace(Y))
    except linalg.LinAlgError:
        try:
            A_inv = linalg.inv(A)
            return float(np.trace(A_inv @ B))
        except linalg.LinAlgError:
            return float(np.trace(np.linalg.pinv(A) @ B))
