"""QR decomposition and solving."""

from __future__ import annotations

import numpy as np
from scipy import linalg


def qr_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b using QR decomposition.

    Args:
        A: Matrix.
        b: Right-hand side.

    Returns:
        Solution x.
    """
    # TODO: Implement QR-based solving
    return linalg.lstsq(A, b)[0]
