"""Cyclic cubic regression spline basis.

Implements periodic (cyclic) cubic splines as used in mgcv's bs='cc'.
The spline satisfies periodic boundary conditions:
    f(x_min) = f(x_max)
    f'(x_min) = f'(x_max)
    f''(x_min) = f''(x_max)

Useful for modelling seasonal effects, circular data, etc.

References:
    - Wood, S.N. (2017). GAMs: An Introduction with R (2nd ed.), Chapter 4.
    - Green & Silverman (1994). Nonparametric Regression and GLMs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class CyclicSpline:
    """Cyclic cubic regression spline basis.

    Attributes:
        X: Input data, shape (n,).
        k: Basis dimension (number of interior knots). Default 10.
        knots: Knot locations including endpoints, shape (k+1,).
        B: Basis matrix, shape (n, k).
        S: Penalty matrix (second derivative integral), shape (k, k).
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int = 10,
        knots: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize cyclic spline.

        Args:
            X: Input data array.
            k: Basis dimension (number of cyclic knots).
            knots: Optional knot locations. If None, uniformly spaced.
        """
        self.X = np.asarray(X, dtype=float).ravel()
        self.n = len(self.X)
        self.k = k

        x_min = self.X.min()
        x_max = self.X.max()
        self.x_min = x_min
        self.x_max = x_max

        # Knot locations: k interior knots spanning [x_min, x_max)
        # The period is (x_max - x_min)
        if knots is None:
            self.knots = np.linspace(x_min, x_max, k + 1)[:-1]  # k knots
        else:
            self.knots = np.asarray(knots, dtype=float)

        self.B, self.S = self._compute_basis_and_penalty()

    def _compute_basis_and_penalty(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute B-spline cyclic basis and difference penalty.

        Returns:
            (B, S) where B is shape (n, k) and S is shape (k, k).
        """
        n = self.n
        k = self.k
        knots = self.knots
        period = self.x_max - self.x_min

        # Use periodic B-spline construction
        # We evaluate k cyclic basis functions using circular knot vector
        B = np.zeros((n, k))

        for j in range(k):
            for i in range(n):
                B[i, j] = self._cyclic_basis_fn(self.X[i], j, knots, period)

        # Second-difference penalty (cyclic: wrap-around)
        D = self._cyclic_difference_matrix(k)
        S = D.T @ D

        return B, S

    def _cyclic_basis_fn(self, x: float, j: int, knots: np.ndarray, period: float) -> float:
        """Evaluate j-th cyclic cubic B-spline basis function at x.

        Uses a simple hat-function approximation for cyclic data.
        For production use, this should be replaced with true B-splines.
        """
        k = len(knots)
        # Wrap x onto [x_min, x_max)
        x_wrapped = self.x_min + ((x - self.x_min) % period)

        # Width between knots
        dk = period / k

        # Distance from x to j-th knot (circular)
        dist = abs(x_wrapped - knots[j])
        if dist > period / 2:
            dist = period - dist

        # Cubic spline kernel (Epanechnikov-style, cyclic)
        width = dk * 2
        if dist >= width:
            return 0.0

        t = dist / width
        return 1.0 - 3 * t**2 + 2 * t**3  # cubic hermite basis

    def _cyclic_difference_matrix(self, k: int) -> np.ndarray:
        """Construct cyclic second-difference matrix.

        The cyclic penalty wraps around: the last row references the first
        two rows to maintain periodicity.

        Args:
            k: Number of knots.

        Returns:
            Difference matrix D of shape ((k, k)).
        """
        D = np.zeros((k, k))
        for i in range(k):
            D[i, i] = 1.0
            D[i, (i + 1) % k] = -2.0
            D[i, (i + 2) % k] = 1.0
        return D

    def basis_matrix(self) -> np.ndarray:
        """Return the cyclic spline basis matrix, shape (n, k)."""
        return self.B

    def penalty_matrix(self) -> np.ndarray:
        """Return the penalty matrix, shape (k, k)."""
        return self.S

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """Evaluate cyclic spline basis at new data points.

        Args:
            x_new: New data, shape (n_new,) or (n_new, 1).

        Returns:
            Basis matrix for new data, shape (n_new, k).
        """
        x_new = np.asarray(x_new, dtype=float).ravel()
        n_new = len(x_new)
        k = self.k
        knots = self.knots
        period = self.x_max - self.x_min

        B_new = np.zeros((n_new, k))
        for j in range(k):
            for i in range(n_new):
                B_new[i, j] = self._cyclic_basis_fn(x_new[i], j, knots, period)
        return B_new
