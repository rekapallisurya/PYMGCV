"""Trace computation utilities."""

from __future__ import annotations

import numpy as np


def trace_product(A: np.ndarray, B: np.ndarray) -> float:
    """Compute trace(A @ B) efficiently.

    Args:
        A: Matrix.
        B: Matrix.

    Returns:
        trace(A @ B).
    """
    return float(np.trace(A @ B))
