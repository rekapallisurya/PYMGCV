"""P-splines (Penalized B-splines) basis.

P-splines combine B-splines with discrete difference penalties.

Theory:
    Instead of penalizing derivatives (as in cubic splines), P-splines
    use a discrete difference penalty on adjacent coefficients:

        Penalty = λ Σⱼ (Δⁿ βⱼ)²

    where Δⁿ is the n-th order difference operator.

Benefits:
    - Computationally simple
    - B-spline basis + discrete penalty (no derivative computation)
    - Good flexibility in mixing basis order and penalty order
    - Often comparable performance to other methods

References:
    - Eilers, P.H.C. and B. Marx (1996). Flexible smoothing with
      B-splines and penalties. Statistical Science, 11(2), 89-121.
    - Wood, S.N. (2017). GAMs: An Introduction with R.

Module exports:
    - PSplineBasis: Main P-spline basis class
    - difference_penalty: Create difference penalty matrix
"""

from __future__ import annotations

import numpy as np


class PSplineBasis:
    """P-spline (penalized B-spline) basis.

    Combines B-spline basis with discrete difference penalties.

    Attributes:
        X: Input data.
        k: Number of basis functions.
        order: B-spline order.
        diff_order: Difference penalty order (typically 1 or 2).
        basis_matrix: Computed basis matrix.
        penalty_matrix: Computed penalty matrix.
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int = 10,
        order: int = 4,
        diff_order: int = 2,
    ) -> None:
        """Initialize P-spline basis.

        Args:
            X: Input data.
            k: Number of basis functions.
            order: B-spline order (degree + 1).
            diff_order: Difference penalty order (1=first diff, 2=second diff).
        """
        self.X = np.asarray(X, dtype=np.float64).ravel()
        self.n = len(self.X)
        self.k = k
        self.order = order
        self.diff_order = diff_order

        # Import B-spline basis
        from pymgcv.smooth.bspline import BSplineBasis

        self.bspline = BSplineBasis(self.X, k=k, order=order)
        self.basis_matrix = self.bspline.basis_matrix

        # Create difference penalty
        self.penalty_matrix = difference_penalty(k, order=diff_order)

    def summary(self) -> str:
        """Summary of P-spline basis."""
        lines = [
            "P-spline (Penalized B-spline) Basis",
            "=" * 45,
            f"Observations: {self.n}",
            f"Basis functions (k): {self.k}",
            f"B-spline order: {self.order}",
            f"Difference penalty order: {self.diff_order}",
        ]
        return "\n".join(lines)


def difference_penalty(k: int, order: int = 2) -> np.ndarray:
    """Create difference penalty matrix for P-splines.

    The penalty matrix corresponds to:
        Penalty = λ Σⱼ (Δⁿ βⱼ)²

    where Δⁿ is the n-th order difference operator.

    For order=1: Δ βⱼ = βⱼ - βⱼ₋₁
    For order=2: Δ² βⱼ = (βⱼ - βⱼ₋₁) - (βⱼ₋₁ - βⱼ₋₂)

    Args:
        k: Number of basis functions (size of penalty matrix).
        order: Difference order (1 or 2, typically).

    Returns:
        Penalty matrix, shape (k, k), symmetric positive semi-definite.
    """
    if order == 1:
        # First difference: Δ βⱼ = βⱼ - βⱼ₋₁
        D = np.diff(np.eye(k), axis=0)  # shape (k-1, k)
        S = D.T @ D  # shape (k, k)
    elif order == 2:
        # Second difference: Δ² βⱼ = (βⱼ - βⱼ₋₁) - (βⱼ₋₁ - βⱼ₋₂)
        D1 = np.diff(np.eye(k), axis=0)  # shape (k-1, k)
        D2 = np.diff(D1, axis=0)  # shape (k-2, k)
        S = D2.T @ D2  # shape (k, k)
    elif order == 3:
        # Third difference
        D1 = np.diff(np.eye(k), axis=0)
        D2 = np.diff(D1, axis=0)
        D3 = np.diff(D2, axis=0)
        S = D3.T @ D3
    else:
        raise ValueError(f"order must be 1, 2, or 3, got {order}")

    return S


def compare_pspline_bases(
    X: np.ndarray,
    bases: list[dict] | None = None,
) -> dict:
    """Compare different P-spline bases.

    Args:
        X: Input data.
        bases: List of dicts with keys 'k', 'order', 'diff_order'.
               If None, uses default configurations.

    Returns:
        Dictionary with comparison results.
    """
    if bases is None:
        bases = [
            {"k": 10, "order": 3, "diff_order": 1},
            {"k": 10, "order": 4, "diff_order": 2},
            {"k": 15, "order": 4, "diff_order": 2},
        ]

    results = {}
    for i, config in enumerate(bases):
        try:
            pspline = PSplineBasis(
                X, k=config["k"], order=config["order"], diff_order=config["diff_order"]
            )
            key = (
                f"Config{i+1}(k={config['k']},order={config['order']},diff={config['diff_order']})"
            )
            results[key] = {
                "basis_shape": pspline.basis_matrix.shape,
                "penalty_rank": np.linalg.matrix_rank(pspline.penalty_matrix),
                "summary": pspline.summary(),
            }
        except Exception as e:
            results[f"Config{i+1}"] = f"Error: {str(e)}"

    return results
