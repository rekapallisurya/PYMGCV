"""Eigen decomposition utilities."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def symmetric_eigen(A: np.ndarray, k: int | None = None):
    """Compute eigendecomposition of symmetric matrix.

    Args:
        A: Symmetric matrix.
        k: Number of eigenvalues to compute (if None, all).

    Returns:
        (eigenvalues, eigenvectors)
    """
    # TODO: Implement partial eigendecomposition if k provided
    eigenvalues, eigenvectors = linalg.eigh(A)
    return eigenvalues, eigenvectors
