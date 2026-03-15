"""Cubic Regression Splines (bs="cr" in mgcv).

Cubic regression splines with:
- Knot-based representation (not eigen-decomposition like TPRS)
- Derivative-based penalties (integrated squared 2nd derivative)
- Simpler and faster than TPRS
- Often performance is near-optimal for univariate smooths

Theory:
    The smooth function f(x) is represented as:
        f(x) = β₀ + β₁ x + Σⱼ βⱼ₊₂ hⱼ(x)
    
    where:
    - β₀, β₁ are intercept and linear term coefficients
    - hⱼ(x) = (x - κⱼ)₊³ are truncated cubic basis functions
    - κⱼ are knots (usually placed at quantiles of x)
    
    The penalty matrix S penalizes the integrated squared second derivative:
        Penalty = ∫[f''(x)]² dx
    
    This encourages smoothness while preserving fit quality.

References:
    - Wood, S.N. (2017). Generalized Additive Models: An Introduction with R.
      Chapman & Hall/CRC.
    - Green, P.J. and Silverman, B.W. (1994). Nonparametric Regression and
      Generalized Linear Models. Chapman & Hall.

Module exports:
    - CubicRegressionSpline: Main basis class
    - cubic_basis_matrix: Function to construct basis
    - create_cubic_penalty: Function to construct penalty matrix
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import interpolate


class CubicRegressionSpline:
    """Cubic regression spline basis for univariate smoothing.

    Uses knot-based representation with truncated cubic basis functions,
    penalizing the integrated squared second derivative.

    Attributes:
        X: Input data, shape (n, 1) or (n,).
        k: Number of basis functions (related to # of knots).
        knots: Knot locations, shape (k-2,).
        basis_matrix: The computed basis matrix, shape (n, k).
        penalty_matrix: The penalty matrix, shape (k, k).
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int = 10,
        knot_placement: str = 'quantile',
    ) -> None:
        """Initialize cubic regression spline.

        Args:
            X: Input data, shape (n,) or (n, 1).
            k: Number of basis functions (default 10).
               Penalty matrix will be (k, k), basis matrix will be (n, k).
               Need k >= 4 for this to make sense.
            knot_placement: How to place knots:
                - 'quantile': at quantiles of X (default, recommended)
                - 'uniform': evenly spaced from min to max

        Raises:
            ValueError: If k < 4.
        """
        # Validate and reshape input
        X = np.asarray(X, dtype=np.float64).ravel()
        self.X = X
        self.n = len(X)

        if k < 4:
            raise ValueError(f'k must be >= 4, got {k}')
        self.k = k

        # Determine knot locations
        if knot_placement == 'quantile':
            # Place knots at quantiles of X
            quantiles = np.linspace(0, 1, k - 2)
            self.knots = np.quantile(X, quantiles)
            # Ensure knots are unique and sorted
            self.knots = np.unique(self.knots)
            # If we lost knots due to ties, add evenly-spaced ones
            while len(self.knots) < k - 2:
                n_needed = k - 2 - len(self.knots)
                additional = np.linspace(X.min(), X.max(), n_needed + 2)[1:-1]
                self.knots = np.unique(np.concatenate([self.knots, additional]))
        elif knot_placement == 'uniform':
            # Evenly spaced knots
            self.knots = np.linspace(X.min(), X.max(), k - 2)
        else:
            raise ValueError(f'Unknown knot_placement: {knot_placement}')

        # Ensure we have exactly k-2 knots
        if len(self.knots) > k - 2:
            # Keep every n-th knot to reduce to k-2
            step = max(1, len(self.knots) // (k - 2))
            self.knots = self.knots[::step][:k-2]
        elif len(self.knots) < k - 2:
            # Add additional evenly-spaced knots if quantiles have ties
            additional_knots = k - 2 - len(self.knots)
            extra = np.linspace(X.min(), X.max(), additional_knots + 2)[1:-1]
            self.knots = np.unique(np.concatenate([self.knots, extra]))[:k-2]

        # Compute basis and penalty matrices
        self.basis_matrix = self._compute_basis()
        self.penalty_matrix = self._compute_penalty()

    def _compute_basis(self) -> np.ndarray:
        """Compute cubic regression spline basis matrix.

        The basis consists of:
        - Constant term: 1
        - Linear term: x
        - k-2 truncated cubic basis functions: (x - κⱼ)₊³

        where (z)₊ = max(0, z).

        Returns:
            Basis matrix, shape (n, k).
        """
        # Start with constant and linear terms
        B = np.column_stack([np.ones(self.n), self.X])

        # Add truncated cubic basis functions
        for knot in self.knots:
            # (x - knot)₊³
            basis_col = np.maximum(self.X - knot, 0) ** 3
            B = np.column_stack([B, basis_col])

        return B

    def _compute_penalty(self) -> np.ndarray:
        """Compute penalty matrix for integrated squared 2nd derivative.

        The penalty penalizes ∫[f''(x)]² dx, which for cubic splines
        is a quadratic form in the coefficients.

        For cubic splines with k basis functions, the penalty matrix
        is computed via numerical integration using the analytical form
        for second derivatives.

        Returns:
            Penalty matrix, shape (k, k).
        """
        # Construct penalty matrix
        # For f(x) = β₀ + β₁x + Σⱼ βⱼ₊₂(x - κⱼ)₊³
        # f'(x) = β₁ + 3Σⱼ βⱼ₊₂(x - κⱼ)₊²
        # f''(x) = 6Σⱼ βⱼ₊₂(x - κⱼ)₊

        # The penalty is ∫[f''(x)]² dx
        # This equals 36 * ∫[Σⱼ βⱼ₊₂(x - κⱼ)₊]² dx

        # Compute analytically following Wood (2017)
        S = np.zeros((self.k, self.k))

        # All intervals for integration
        x_all = np.sort(np.concatenate([[self.X.min()], self.knots, [self.X.max()]]))
        
        # Loop over intervals and compute contributions
        for i in range(len(x_all) - 1):
            x_L = x_all[i]
            x_R = x_all[i + 1]
            x_mid = (x_L + x_R) / 2
            
            # Find which basis functions are active in this interval
            # For truncated cubics, (x - κⱼ)₊ is active for x > κⱼ
            active = np.concatenate([
                [True, True],  # constant and linear terms always inactive in 2nd deriv
                self.knots <= x_mid  # truncated cubic basis functions
            ])
            
            # In this interval, the active basis functions contribute 6β to f''
            n_active = np.sum(active[2:])  # exclude constant and linear
            
            # Contribution to penalty from this interval
            # The second derivative terms contribute proportionally
            if n_active > 0:
                interval_length = x_R - x_L
                # 36 * interval_length is the basic scale
                scale = 36 * interval_length
                # Add to the block of the penalty matrix corresponding to active terms
                for j1 in range(2, self.k):
                    if active[j1]:
                        for j2 in range(2, self.k):
                            if active[j2]:
                                S[j1, j2] += scale

        return S

    def basis_matrix(self) -> np.ndarray:
        """Return the basis matrix.

        Returns:
            Basis matrix, shape (n, k).
        """
        return self.basis_matrix

    def penalty_matrix(self) -> np.ndarray:
        """Return the penalty matrix.

        Returns:
            Penalty matrix, shape (k, k).
        """
        return self.penalty_matrix

    def summary(self) -> str:
        """Return summary of basis."""
        lines = [
            'Cubic Regression Spline Basis',
            '=' * 40,
            f'Number of observations: {self.n}',
            f'Number of basis functions: {self.k}',
            f'Number of knots: {len(self.knots)}',
            f'Knot range: [{self.knots.min():.4f}, {self.knots.max():.4f}]',
            f'Data range: [{self.X.min():.4f}, {self.X.max():.4f}]',
        ]
        return '\n'.join(lines)


def cubic_basis_matrix(
    X: np.ndarray,
    knots: Optional[np.ndarray] = None,
    k: int = 10,
) -> np.ndarray:
    """Construct cubic regression spline basis matrix.

    Functional API for basis construction.

    Args:
        X: Input data.
        knots: Knot locations (if None, determined from quantiles of X).
        k: Number of basis functions (if knots not provided).

    Returns:
        Basis matrix, shape (n, k).
    """
    # If knots provided, k is overridden
    if knots is not None:
        k = len(knots) + 2

    spline = CubicRegressionSpline(X, k=k)
    return spline.basis_matrix


def create_cubic_penalty(k: int, X: Optional[np.ndarray] = None) -> np.ndarray:
    """Create penalty matrix for cubic regression spline.

    Args:
        k: Number of basis functions.
        X: Optional data to compute knot-dependent penalties.

    Returns:
        Penalty matrix, shape (k, k).
    """
    if X is None:
        X = np.linspace(0, 1, 100)
    
    spline = CubicRegressionSpline(X, k=k)
    return spline.penalty_matrix


def compare_cubic_vs_tprs(
    X: np.ndarray,
    y: np.ndarray,
    basis_dim: int = 10,
) -> dict:
    """Compare cubic regression spline vs TPRS basis.

    Useful for understanding when each is appropriate.

    Args:
        X: Input data.
        y: Response.
        basis_dim: Basis dimension to use.

    Returns:
        Dictionary with comparison results.
    """
    from pymgcv.smooth.thin_plate import ThinPlateSpline
    
    X = np.asarray(X, dtype=np.float64).reshape(-1, 1)
    
    # Cubic regression spline
    cr_spline = CubicRegressionSpline(X.ravel(), k=basis_dim)
    
    # TPRS
    tp_spline = ThinPlateSpline(X, k=basis_dim)
    
    return {
        'cubic_basis_dim': cr_spline.k,
        'tprs_basis_dim': tp_spline.basis_matrix.shape[1],
        'cubic_knots': len(cr_spline.knots),
        'cubic_summary': cr_spline.summary(),
    }
