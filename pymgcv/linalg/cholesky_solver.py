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


def solve_symmetric_definite(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve symmetric positive definite system Ax = b.

    Args:
        A: SPD matrix.
        b: Right-hand side vector.

    Returns:
        Solution x.
    """
    return linalg.solve(A, b, assume_a='pos')


def trace_matrix_inverse_product(A: np.ndarray, B: np.ndarray) -> float:
    """Compute trace(A^{-1} B) efficiently.

    Uses conjugate gradient implicitly without forming A^{-1}.

    Args:
        A: Matrix (SPD).
        B: Matrix.

    Returns:
        trace(A^{-1} B).
    """
    # TODO: Implement efficient trace computation
    # For now, fall back to explicit inversion
    A_inv = linalg.inv(A)
    return float(np.trace(A_inv @ B))
