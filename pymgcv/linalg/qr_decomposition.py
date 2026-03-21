"""QR decomposition and solving."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def qr_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b using QR decomposition.

    Uses an economy (thin) QR factorisation A = Q R, then solves
    R x = Q' b via back-substitution.  This is numerically stable
    and equivalent to the least-squares solution when A is full-rank.
    Falls back to scipy lstsq when A is rank-deficient.

    Args:
        A: Matrix of shape (m, n), m >= n.
        b: Right-hand side vector of shape (m,) or matrix (m, k).

    Returns:
        Solution x of shape (n,) / (n, k).
    """
    Q, R = linalg.qr(A, mode='economic')
    diag_R = np.abs(np.diag(R))
    tol = max(float(diag_R[0]), 1.0) * A.shape[1] * np.finfo(float).eps * 100
    if np.all(diag_R > tol):
        # Full-rank: back-substitute
        return linalg.solve_triangular(R, Q.T @ b, lower=False)
    # Rank-deficient fallback
    return linalg.lstsq(A, b)[0]
