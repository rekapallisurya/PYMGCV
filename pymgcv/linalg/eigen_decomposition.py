"""Eigen decomposition utilities."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def symmetric_eigen(A: np.ndarray, k: int | None = None):
    """Compute eigendecomposition of a real symmetric matrix.

    When k is given, returns only the k largest-magnitude eigenpairs
    using scipy's subset_by_index for efficiency (avoids full decomposition
    on large matrices).

    Args:
        A: Real symmetric matrix of shape (n, n).
        k: Number of eigenpairs to return (top-k by magnitude).
           If None, all n eigenpairs are returned.

    Returns:
        (eigenvalues, eigenvectors) – both sorted ascending.
    """
    n = A.shape[0]
    if k is None or k >= n:
        eigenvalues, eigenvectors = linalg.eigh(A)
        return eigenvalues, eigenvectors

    # Clamp k to valid range
    k = max(1, min(k, n))
    # subset_by_index selects indices [lo, hi] (0-based, inclusive)
    lo = n - k
    hi = n - 1
    eigenvalues, eigenvectors = linalg.eigh(A, subset_by_index=(lo, hi))
    return eigenvalues, eigenvectors
