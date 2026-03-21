"""Native math backend for pymgcv.

This module provides two acceleration layers:
1. A compiled C extension for custom kernels.
2. Direct LAPACK calls from SciPy (`dposv`) backed by Fortran libraries.

Both paths include NumPy/SciPy fallbacks for portability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg
from scipy.linalg import lapack

try:
    from . import _native_c

    HAS_C_ENGINE = True
except ImportError:
    _native_c = None
    HAS_C_ENGINE = False


def backend_info() -> dict[str, Any]:
    """Report active native backend capabilities."""
    return {
        "c_engine": HAS_C_ENGINE,
        "fortran_lapack": hasattr(lapack, "dposv"),
    }


def col_squared_norms(a: np.ndarray) -> np.ndarray:
    """Compute squared L2 norm of each column.

    Uses the C extension when available; otherwise falls back to NumPy.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("Expected a 2D array")

    if HAS_C_ENGINE and _native_c is not None:
        return _native_c.col_squared_norms(a)

    return np.sum(a * a, axis=0)


def solve_spd_fortran(a: np.ndarray, b: np.ndarray, *, lower: bool = True) -> np.ndarray:
    """Solve Ax=b for SPD A using LAPACK `dposv` (Fortran routine).

    Falls back to SciPy's high-level SPD solver when low-level LAPACK is
    unavailable or returns a failure code.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be a square 2D matrix")

    if b.ndim not in (1, 2):
        raise ValueError("b must be 1D or 2D")

    dposv = getattr(lapack, "dposv", None)
    if dposv is None:
        return linalg.solve(a, b, assume_a="pos")

    _, x, info = dposv(a, b, lower=1 if lower else 0, overwrite_a=0, overwrite_b=0)
    if info == 0:
        return x

    return linalg.solve(a, b, assume_a="pos")
