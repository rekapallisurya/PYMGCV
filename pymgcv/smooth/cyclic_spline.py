"""Cyclic spline basis types.

Provides:
  - CyclicSpline      (bs='cc') — periodic cubic regression spline
  - CyclicPSpline     (bs='cp') — periodic P-spline with difference penalty

Both use true periodic B-spline basis functions constructed by wrapping
a standard B-spline knot vector. The penalty differs:
  - cc: integrated squared second derivative (continuous penalty)
  - cp: m-th order cyclic difference penalty (discrete, Eilers-Marx)

References:
    - Wood, S.N. (2017). GAMs: An Introduction with R (2nd ed.), Chapter 4.
    - Eilers, P.H.C. & Marx, B.D. (1996). Flexible smoothing with B-splines
      and penalties. Statistical Science, 11(2), 89-121.
    - de Boor, C. (1978). A Practical Guide to Splines.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _periodic_bspline_design(
    x: np.ndarray,
    k: int,
    degree: int = 3,
    x_min: float | None = None,
    x_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build periodic B-spline design matrix.

    Places k uniformly-spaced knots on [x_min, x_max) and constructs a
    periodic B-spline basis by wrapping the extended knot vector.

    Parameters
    ----------
    x : array, shape (n,)
        Data points.
    k : int
        Number of basis functions (= number of periodic knots).
    degree : int
        B-spline degree (default 3 = cubic).
    x_min, x_max : float, optional
        Period boundaries. Default: data range.

    Returns
    -------
    B : array, shape (n, k)
        Periodic B-spline design matrix.
    knot_pos : array, shape (k,)
        The k uniformly-spaced knot positions.
    t : array
        Full extended knot vector.
    period : float
        Period length.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x_min is None:
        x_min = float(x.min())
    if x_max is None:
        x_max = float(x.max())
    period = x_max - x_min
    if period <= 0:
        period = 1.0
    h = period / k

    # Wrap x into [x_min, x_max)
    x_w = x_min + ((x - x_min) % period)

    # k uniformly spaced knot positions
    knot_pos = x_min + np.arange(k) * h

    # Extended knot vector: degree extra on each side for wrapping
    t = np.concatenate(
        [
            knot_pos[-degree:] - period,  # degree knots left of x_min
            knot_pos,  # k interior knots
            knot_pos[: degree + 1] + period,  # degree+1 knots right of x_max
        ]
    )

    n_ext = len(t) - degree - 1  # = k + degree

    # Build extended design matrix using scipy
    try:
        B_sparse = BSpline.design_matrix(x_w, t, degree)
        B_ext = B_sparse.toarray() if hasattr(B_sparse, "toarray") else np.asarray(B_sparse)
    except (AttributeError, TypeError):
        B_ext = np.zeros((len(x_w), n_ext))
        for j in range(n_ext):
            c = np.zeros(n_ext)
            c[j] = 1.0
            spl = BSpline(t, c, degree, extrapolate=True)
            B_ext[:, j] = spl(x_w)

    # Wrap overflow columns back to the beginning (periodic identification)
    B = B_ext[:, :k].copy()
    overflow = n_ext - k  # = degree
    for j in range(overflow):
        B[:, j] += B_ext[:, k + j]

    return B, knot_pos, t, period


def _periodic_integral_penalty(
    t: np.ndarray,
    k: int,
    degree: int,
    period: float,
    x_min: float,
    deriv_order: int = 2,
) -> np.ndarray:
    """Integrated squared derivative penalty for periodic B-splines.

    S_ij = integral B_i^(m)(x) B_j^(m)(x) dx  over one period.
    """
    n_ext = len(t) - degree - 1
    if degree < deriv_order:
        return np.zeros((k, k))

    x_max = x_min + period
    gl_pts, gl_wts = leggauss(max(degree * 2, 10))

    # Integration over sub-intervals between unique knots in [x_min, x_max]
    unique_t = np.unique(t)
    unique_t = unique_t[(unique_t >= x_min - 1e-10) & (unique_t <= x_max + 1e-10)]
    if len(unique_t) < 2:
        unique_t = np.array([x_min, x_max])

    S_ext = np.zeros((n_ext, n_ext))
    for left, right in zip(unique_t[:-1], unique_t[1:]):
        if right - left < 1e-14:
            continue
        mid = 0.5 * (left + right)
        half = 0.5 * (right - left)
        x_gl = mid + half * gl_pts
        w_gl = half * gl_wts

        # Evaluate m-th derivative of each B-spline at quadrature nodes
        Bd = np.zeros((len(x_gl), n_ext))
        for j in range(n_ext):
            c = np.zeros(n_ext)
            c[j] = 1.0
            spl = BSpline(t, c, degree, extrapolate=False)
            spl_d = spl.derivative(deriv_order)
            vals = spl_d(x_gl)
            Bd[:, j] = np.where(np.isfinite(vals), vals, 0.0)

        S_ext += (Bd * w_gl[:, None]).T @ Bd

    # Wrap penalty matrix to periodic form
    S = S_ext[:k, :k].copy()
    overflow = n_ext - k
    for j in range(overflow):
        S[:, j] += S_ext[:k, k + j]
        S[j, :] += S_ext[k + j, :k]
        for jj in range(overflow):
            S[j, jj] += S_ext[k + j, k + jj]

    return 0.5 * (S + S.T)


def _cyclic_diff_penalty(k: int, m: int = 2) -> np.ndarray:
    """Cyclic m-th order difference penalty D'D.

    The difference operator wraps around: row i references columns
    (i, i+1, ..., i+m) mod k, using binomial coefficients.
    """
    from math import comb

    coeffs = [(-1) ** j * comb(m, j) for j in range(m + 1)]
    D = np.zeros((k, k))
    for i in range(k):
        for j_off, c in enumerate(coeffs):
            D[i, (i + j_off) % k] += c
    return D.T @ D


# ---------------------------------------------------------------------------
# CyclicSpline (bs='cc')
# ---------------------------------------------------------------------------


class CyclicSpline:
    """Cyclic cubic regression spline basis (bs='cc').

    Periodic B-spline basis with integrated squared second derivative
    penalty. The spline satisfies periodic boundary conditions:
        f(x_min) = f(x_max),  f'(x_min) = f'(x_max),  f''(x_min) = f''(x_max)

    Attributes
    ----------
    X : array, shape (n,)
        Input data.
    k : int
        Number of basis functions.
    B : array, shape (n, k)
        Basis matrix.
    S : array, shape (k, k)
        Integrated squared second derivative penalty.
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int = 10,
        knots: np.ndarray | None = None,
    ) -> None:
        self.X = np.asarray(X, dtype=float).ravel()
        self.n = len(self.X)
        self.k = k

        self.x_min = float(self.X.min())
        self.x_max = float(self.X.max())

        # Build periodic B-spline basis
        self.B, self.knots, self._t, self._period = _periodic_bspline_design(
            self.X, k, degree=3, x_min=self.x_min, x_max=self.x_max
        )

        # Integrated squared second derivative penalty
        self.S = _periodic_integral_penalty(
            self._t, k, degree=3, period=self._period, x_min=self.x_min, deriv_order=2
        )

    def basis_matrix(self) -> np.ndarray:
        """Return the cyclic spline basis matrix, shape (n, k)."""
        return self.B

    def penalty_matrix(self) -> np.ndarray:
        """Return the penalty matrix, shape (k, k)."""
        return self.S

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """Evaluate cyclic spline basis at new data points."""
        B_new, _, _, _ = _periodic_bspline_design(
            x_new, self.k, degree=3, x_min=self.x_min, x_max=self.x_max
        )
        return B_new


# ---------------------------------------------------------------------------
# CyclicPSpline (bs='cp')
# ---------------------------------------------------------------------------


class CyclicPSpline:
    """Cyclic P-spline basis (bs='cp').

    Periodic B-spline basis with cyclic difference penalty.
    Uses Eilers-Marx approach with wrap-around differences.

    Attributes
    ----------
    X : array, shape (n,)
        Input data.
    k : int
        Number of basis functions.
    B : array, shape (n, k)
        Basis matrix.
    S : array, shape (k, k)
        Cyclic difference penalty D'D.
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int = 20,
        knots: np.ndarray | None = None,
        m: int = 2,
        degree: int = 3,
    ) -> None:
        self.X = np.asarray(X, dtype=float).ravel()
        self.n = len(self.X)
        self.k = k
        self.m = m
        self.degree = degree

        self.x_min = float(self.X.min())
        self.x_max = float(self.X.max())

        # Build periodic B-spline basis
        self.B, self.knots, self._t, self._period = _periodic_bspline_design(
            self.X, k, degree=degree, x_min=self.x_min, x_max=self.x_max
        )

        # Cyclic difference penalty
        self.S = _cyclic_diff_penalty(k, m)

    def basis_matrix(self) -> np.ndarray:
        """Return the cyclic P-spline basis matrix, shape (n, k)."""
        return self.B

    def penalty_matrix(self) -> np.ndarray:
        """Return the penalty matrix, shape (k, k)."""
        return self.S

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """Evaluate basis at new data points."""
        B_new, _, _, _ = _periodic_bspline_design(
            x_new, self.k, degree=self.degree, x_min=self.x_min, x_max=self.x_max
        )
        return B_new
