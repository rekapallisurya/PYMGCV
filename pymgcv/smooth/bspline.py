"""B-spline and P-spline basis for GAM smoothing.

Provides:
  - BSplineBasis  (bs='bs')  — integrated-squared-derivative penalty
  - PSplineBasis  (bs='ps')  — Eilers-Marx difference penalty (faster)
  - Shared low-level helpers for knot construction and penalty matrices

References:
    - de Boor, C. (1978). A Practical Guide to Splines.
    - Eilers, P.H.C. & Marx, B.D. (1996). Flexible smoothing with B-splines
      and penalties. Statistical Science, 11(2), 89-121.
    - Wood, S.N. (2017). GAMs: An Introduction with R.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import BSpline
from numpy.polynomial.legendre import leggauss


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _make_knots(x: np.ndarray, k: int, degree: int) -> np.ndarray:
    """Build clamped B-spline knot vector with quantile interior knots.

    Total knots = k + degree + 1.
    """
    n_interior = k - degree - 1
    x_min, x_max = float(x.min()), float(x.max())
    if x_min == x_max:
        x_max = x_min + 1.0
    if n_interior <= 0:
        return np.concatenate([
            np.full(degree + 1, x_min),
            np.full(degree + 1, x_max),
        ])
    q = np.linspace(0, 1, n_interior + 2)[1:-1]
    interior = np.quantile(x, q)
    interior = np.clip(interior, x_min + 1e-12, x_max - 1e-12)
    return np.concatenate([
        np.full(degree + 1, x_min),
        interior,
        np.full(degree + 1, x_max),
    ])


def _bspline_design_matrix(x: np.ndarray, t: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate all B-spline basis functions at each data point.

    Args:
        x: Data values, shape (n,).
        t: Knot vector.
        degree: Polynomial degree.

    Returns:
        Basis matrix B, shape (n, k) where k = len(t) - degree - 1.
    """
    k = len(t) - degree - 1
    # BSpline.design_matrix added in scipy 1.7; fall back for older versions
    try:
        B_sparse = BSpline.design_matrix(x, t, degree)
        B = B_sparse.toarray() if hasattr(B_sparse, 'toarray') else np.asarray(B_sparse)
        return B
    except (AttributeError, TypeError):
        B = np.zeros((len(x), k))
        for i in range(k):
            c = np.zeros(k)
            c[i] = 1.0
            spl = BSpline(t, c, degree, extrapolate=True)
            B[:, i] = spl(x)
        return B


def _diff_matrix(k: int, m: int) -> np.ndarray:
    """m-th order finite-difference matrix D, shape (k-m, k).  S = D.T @ D."""
    D = np.eye(k)
    for _ in range(m):
        D = np.diff(D, axis=0)
    return D


def _bspline_integral_penalty(t: np.ndarray, degree: int, deriv_order: int = 2) -> np.ndarray:
    """Integrated-squared-derivative penalty matrix via Gauss-Legendre quadrature.

    S_ij = ∫ (d^m/dx^m B_i(x)) (d^m/dx^m B_j(x)) dx

    Args:
        t: Knot vector (clamped).
        degree: Polynomial degree.
        deriv_order: Derivative order m (default 2).

    Returns:
        Penalty matrix S, shape (k, k).
    """
    k = len(t) - degree - 1
    if degree < deriv_order:
        return np.zeros((k, k))

    unique_knots = np.unique(t)
    S = np.zeros((k, k))
    gl_pts, gl_wts = leggauss(max(5, degree + 1))

    for left, right in zip(unique_knots[:-1], unique_knots[1:]):
        if right - left < 1e-14:
            continue
        mid = 0.5 * (left + right)
        half = 0.5 * (right - left)
        x_gl = mid + half * gl_pts
        w_gl = half * gl_wts

        # Evaluate m-th derivative of each B-spline at quadrature nodes
        Bd = np.zeros((len(x_gl), k))
        for i in range(k):
            c = np.zeros(k)
            c[i] = 1.0
            spl = BSpline(t, c, degree, extrapolate=False)
            spl_d = spl.derivative(deriv_order)
            vals = spl_d(x_gl)
            Bd[:, i] = np.where(np.isfinite(vals), vals, 0.0)

        S += (Bd * w_gl[:, None]).T @ Bd

    return S


# ---------------------------------------------------------------------------
# BSplineBasis  (bs='bs')
# ---------------------------------------------------------------------------

class BSplineBasis:
    """B-spline basis with integrated-squared-derivative penalty (bs='bs').

    Attributes:
        X: Input data, shape (n,).
        k: Number of basis functions.
        degree: Polynomial degree (default 3 = cubic).
        penorder: Derivative order to penalise (default 2).
        knots: Full clamped knot vector.
        B: Basis matrix, shape (n, k).
        S: Penalty matrix, shape (k, k).
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int = 10,
        degree: int = 3,
        penorder: int = 2,
        # Legacy alias: order = degree + 1
        order: Optional[int] = None,
    ) -> None:
        self.X = np.asarray(X, dtype=float).ravel()
        self.n = len(self.X)
        # Support legacy 'order' parameter
        if order is not None:
            degree = order - 1
        self.k = k
        self.degree = degree
        self.order = degree + 1  # keep old attribute
        self.penorder = penorder

        if k <= degree:
            raise ValueError(f'k={k} must be > degree={degree}')

        self.knots = _make_knots(self.X, k, degree)
        self.B = _bspline_design_matrix(self.X, self.knots, degree)
        self.S = _bspline_integral_penalty(self.knots, degree, penorder)

        # Legacy attribute names kept for backward compat
        self.basis_matrix = self.B
        self.penalty_matrix = self.S

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=float).ravel()
        return _bspline_design_matrix(X_new, self.knots, self.degree)


# ---------------------------------------------------------------------------
# PSplineBasis  (bs='ps')
# ---------------------------------------------------------------------------

class PSplineBasis:
    """P-spline: uniform B-spline basis + discrete difference penalty (bs='ps').

    Eilers & Marx (1996) approach: equally-spaced interior knots + D^T D penalty
    on consecutive B-spline coefficients.  Much faster than BSplineBasis for
    large data because the penalty is a simple banded matrix.

    Attributes:
        X: Input data, shape (n,).
        k: Number of B-spline coefficients.
        degree: Polynomial degree (default 3).
        m: Difference order for penalty (default 2).
        B: Basis matrix, shape (n, k).
        S: P-spline penalty D^T D, shape (k, k).
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int = 20,
        degree: int = 3,
        m: int = 2,
    ) -> None:
        self.X = np.asarray(X, dtype=float).ravel()
        self.n = len(self.X)
        self.k = k
        self.degree = degree
        self.m = m

        if k <= degree:
            raise ValueError(f'k={k} must be > degree={degree}')

        x_min, x_max = float(self.X.min()), float(self.X.max())
        if x_min == x_max:
            x_max = x_min + 1.0
        n_interior = k - degree - 1
        interior = np.linspace(x_min, x_max, n_interior + 2)[1:-1] if n_interior > 0 else np.array([])
        t = np.concatenate([
            np.full(degree + 1, x_min),
            interior,
            np.full(degree + 1, x_max),
        ])
        self.knots = t
        self.B = _bspline_design_matrix(self.X, t, degree)

        D = _diff_matrix(k, m)
        self.S = D.T @ D

    def basis_matrix(self) -> np.ndarray:
        return self.B

    def penalty_matrix(self) -> np.ndarray:
        return self.S

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=float).ravel()
        return _bspline_design_matrix(X_new, self.knots, self.degree)


# ---------------------------------------------------------------------------
# Functional API (backward compat)
# ---------------------------------------------------------------------------

def bspline_basis_matrix(
    X: np.ndarray,
    k: int = 10,
    order: int = 4,
) -> np.ndarray:
    """Construct B-spline basis matrix."""
    return BSplineBasis(X, k=k, order=order).B


def bspline_penalty(k: int, order: int = 4, penorder: int = 2) -> np.ndarray:
    """Create integrated-squared-derivative penalty matrix."""
    degree = order - 1
    x_dummy = np.linspace(0, 1, max(k * 3, 30))
    t = _make_knots(x_dummy, k, degree)
    return _bspline_integral_penalty(t, degree, penorder)

