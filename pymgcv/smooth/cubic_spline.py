"""Cubic Regression Splines (bs='cr') and Cubic Shrinkage Splines (bs='cs').

Natural cubic regression splines with an integrated-squared-second-derivative
penalty.  This is mgcv's default univariate smooth.

Also provides CubicShrinkageSpline (bs='cs') which adds an extra shrinkage
component to the null-space (constant + linear terms) so that unneeded terms
are removed by the penalty.

Theory:
    A natural cubic spline f(x) with knots κ₁ < ... < κₖ is piecewise cubic,
    linear beyond the boundary knots, and has continuous first and second
    derivatives everywhere.

    Penalty: ∫[f''(x)]² dx  — penalises curvature.

    The basis is computed via the standard O'Sullivan et al. construction
    (equivalent to mgcv's cr/cs smooths).

References:
    - Wood, S.N. (2017). GAMs: An Introduction with R, Chapter 4.
    - Green & Silverman (1994). Nonparametric Regression and GLMs.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


def _natural_cubic_spline_basis_and_penalty(
    x: np.ndarray,
    knots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the natural cubic spline basis and penalty.

    The basis has k columns (k = len(knots)):
        col 0  : 1           (intercept — unpenalised)
        col 1  : x           (linear   — unpenalised)
        col 2..k-1 : natural cubic spline functions g_j(x)

    The penalty matrix S has zeros in the top-left 2×2 block and is non-zero
    for the g_j components.

    Reference: Wood (2017) §4.1.2, equations 4.2–4.4.

    Args:
        x: Data values, shape (n,).
        knots: Knot locations, shape (k,).  Should include boundary knots.

    Returns:
        (B, S) where B: (n, k), S: (k, k).
    """
    n = len(x)
    k = len(knots)  # number of knots = number of basis functions
    kappa = np.sort(knots)

    # ---- helper: h_{j}(x) ----
    # Define h_j(x) = (x - κ_j)^3_+ - d_j(x - κ_{k-1})^3_+ + c_j(x - κ_k)^3_+
    # where d_j and c_j are chosen to make the function linear beyond κ_k
    #
    # Wood (2017) eq 4.3:
    #   g_j(x) = [(x-κ_j)^3_+ - (x-κ_k)^3_+ * (κ_k - κ_j)/(κ_k - κ_{k-1})
    #             + (x-κ_{k-1})^3_+ * (κ_k - κ_j)/(κ_k - κ_{k-1})] / (κ_k - κ_{k-1})
    # but the normalisation varies; we use a numerically equivalent form.

    kn = kappa[-1]
    knm1 = kappa[-2]
    h = kn - knm1

    def _hplus(x_val, a):
        return np.maximum(x_val - a, 0.0) ** 3

    # g_j(x) for j = 0 .. k-3  (k-2 spline functions)
    # (adapted from Wood 2003 / Green & Silverman)
    def g_j(x_val, j):
        kj = kappa[j]
        A = _hplus(x_val, kj)
        B = _hplus(x_val, knm1) * ((kn - kj) / h)
        C = _hplus(x_val, kn) * ((knm1 - kj + h) / h)  # = (kn - kj - h)/h
        # Correct formula (Green & Silverman, p. 12):
        c_j = (kn - kj) / h
        d_j = (knm1 - kj) / h  # = c_j - 1
        return A - c_j * _hplus(x_val, knm1) + (c_j - 1) * _hplus(x_val, kn)

    # Build basis: intercept, linear, then k-2 spline functions
    # Total columns: 2 + (k - 2) = k
    B = np.zeros((n, k))
    B[:, 0] = 1.0
    B[:, 1] = x
    for j in range(k - 2):
        B[:, j + 2] = g_j(x, j)

    # ---- Penalty matrix ----
    # S_{ij} = ∫ g_i''(x) g_j''(x) dx  for i,j = 0..k-3
    # g''_j(x) = 6(x - κ_j)_+ - 6 c_j (x - κ_{k-1})_+ + 6(c_j - 1)(x - κ_k)_+
    # The integration domain is [κ_1, κ_k].

    # Integration breakpoints: all unique knots
    all_pts = np.unique(kappa)
    gl_pts, gl_wts = leggauss(5)

    S = np.zeros((k, k))
    for left, right in zip(all_pts[:-1], all_pts[1:]):
        if right - left < 1e-14:
            continue
        mid = 0.5 * (left + right)
        half = 0.5 * (right - left)
        xq = mid + half * gl_pts
        wq = half * gl_wts

        # Evaluate g_j'' at quadrature points
        G2 = np.zeros((len(xq), k - 2))
        for j in range(k - 2):
            kj = kappa[j]
            c_j = (kn - kj) / h
            G2[:, j] = (
                6.0 * np.maximum(xq - kj, 0.0)
                - 6.0 * c_j * np.maximum(xq - knm1, 0.0)
                + 6.0 * (c_j - 1) * np.maximum(xq - kn, 0.0)
            )

        # S[2:, 2:] += ∫ G2_i G2_j dx
        S[2:, 2:] += (G2 * wq[:, None]).T @ G2

    return B, S


class CubicRegressionSpline:
    """Natural cubic regression spline basis (bs='cr').

    Attributes:
        X: Input data, shape (n,).
        k: Number of basis functions (= number of knots).
        knots: Knot locations, shape (k,).
        B: Basis matrix, shape (n, k).
        S: Penalty matrix, shape (k, k).  Zeros in rows/cols 0–1 (null space).
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int = 10,
        knot_placement: str = "quantile",
        knots: np.ndarray | None = None,
    ) -> None:
        X = np.asarray(X, dtype=float).ravel()
        if len(X) < 3:
            raise ValueError("Need at least 3 observations")
        if k < 4:
            raise ValueError("k must be >= 4")

        self.X = X
        self.n = len(X)
        self.k = k

        if knots is not None:
            self.knots = np.sort(np.asarray(knots, dtype=float))
            # If custom knots are interior only, add boundary knots
            if len(self.knots) < k:
                self.knots = np.concatenate(
                    [
                        [X.min()],
                        self.knots,
                        [X.max()],
                    ]
                )
        else:
            if knot_placement == "quantile":
                qs = np.linspace(0, 1, k)
                self.knots = np.unique(np.quantile(X, qs))
                # Ensure we have exactly k knots
                while len(self.knots) < k:
                    extra = np.linspace(X.min(), X.max(), k - len(self.knots) + 2)[1:-1]
                    self.knots = np.unique(np.concatenate([self.knots, extra]))
                self.knots = self.knots[:k]
            else:
                self.knots = np.linspace(X.min(), X.max(), k)

        self.B, self.S = _natural_cubic_spline_basis_and_penalty(X, self.knots)
        # Legacy attribute names
        self.basis_matrix = self.B
        self.penalty_matrix = self.S

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=float).ravel()
        B_new, _ = _natural_cubic_spline_basis_and_penalty(X_new, self.knots)
        return B_new

    def summary(self) -> str:
        return (
            f"CubicRegressionSpline n={self.n} k={self.k} "
            f"knots=[{self.knots[0]:.4f}, {self.knots[-1]:.4f}]"
        )


class CubicShrinkageSpline:
    """Cubic shrinkage spline (bs='cs').

    Like CubicRegressionSpline but adds a small ridge penalty to the null-space
    (intercept + linear term) so they can be shrunk to zero.

    Attributes:
        shrink_factor: Shrinkage weight added to S[0,0] and S[1,1].
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int = 10,
        shrink_factor: float = 1e-4,
    ) -> None:
        self._cr = CubicRegressionSpline(X, k=k)
        self.X = self._cr.X
        self.n = self._cr.n
        self.k = self._cr.k
        self.knots = self._cr.knots
        self.B = self._cr.B
        self.S = self._cr.S.copy()
        # Add shrinkage penalty to null-space
        self.S[0, 0] += shrink_factor
        self.S[1, 1] += shrink_factor
        # Legacy attrs
        self.basis_matrix = self.B
        self.penalty_matrix = self.S

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        return self._cr.predict(X_new)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def cubic_basis_matrix(
    X: np.ndarray,
    k: int = 10,
    knots: np.ndarray | None = None,
) -> np.ndarray:
    spline = CubicRegressionSpline(X, k=k, knots=knots)
    return spline.B


def create_cubic_penalty(k: int, X: np.ndarray | None = None) -> np.ndarray:
    if X is None:
        X = np.linspace(0, 1, max(k * 3, 50))
    return CubicRegressionSpline(X, k=k).S
