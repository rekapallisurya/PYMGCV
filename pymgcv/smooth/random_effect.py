"""Random effect smooth basis (bs='re').

Implements random effects as smooth terms, equivalent to mgcv's bs='re'.
For a factor variable, this creates a ridge penalty (identity) on group means.
Equivalent to including a random intercept per level.

The penalty is just an identity matrix scaled by the number of levels.
The smoothing parameter lambda controls the variance ratio 1/lambda ~ sigma_b^2/sigma_e^2.

References:
    - Wood (2017). GAMs: An Introduction with R, Section 6.6.
    - Fahrmeir et al. (2013). Regression, Chapter 5.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class RandomEffect:
    """Random effect basis for factor or continuous random effects.

    For a factor variable with k levels, creates an indicator (dummy) basis
    matrix with an identity penalty (ridge regression = random effect).

    For a continuous variable, creates a scaled basis with identity penalty.

    Attributes:
        X: Input data (factor or continuous).
        levels: Unique levels if factor variable.
        k: Number of random effect levels (= basis dimension).
        B: Basis matrix, shape (n, k).
        S: Identity penalty matrix, shape (k, k).
    """

    def __init__(
        self,
        X: np.ndarray | pd.Series,
        k: Optional[int] = None,
    ) -> None:
        """Initialize random effect basis.

        Args:
            X: Factor or continuous variable.
            k: Number of levels (auto-detected for factors).
        """
        if isinstance(X, pd.Series):
            X_vals = X.values
        else:
            X_vals = np.asarray(X)

        self.n = len(X_vals)

        # Detect if categorical/factor
        if X_vals.dtype.kind in ('U', 'O', 'S'):
            # String/object → factor
            self.levels = np.unique(X_vals)
            self.k = len(self.levels)
            self.is_factor = True
            self.B = self._build_factor_basis(X_vals)
        elif k is not None:
            # Explicit k for continuous
            self.levels = None
            self.k = k
            self.is_factor = False
            self.B = self._build_continuous_basis(X_vals, k)
        else:
            # Try to detect level count
            unique_vals = np.unique(X_vals)
            self.levels = unique_vals
            self.k = len(unique_vals)
            self.is_factor = True
            self.B = self._build_factor_basis(X_vals.astype(str))

        # Identity penalty = ridge regression
        self.S = np.eye(self.k)

    def _build_factor_basis(self, X: np.ndarray) -> np.ndarray:
        """Build indicator matrix for factor variable.

        Returns:
            Matrix of shape (n, k) with 0/1 entries.
        """
        k = len(self.levels)
        B = np.zeros((self.n, k))
        level_map = {lv: i for i, lv in enumerate(self.levels)}

        for i, val in enumerate(X):
            idx = level_map.get(str(val))
            if idx is not None:
                B[i, idx] = 1.0
        return B

    def _build_continuous_basis(self, X: np.ndarray, k: int) -> np.ndarray:
        """Build polynomial basis for continuous random effect.

        Args:
            X: Continuous variable.
            k: Basis dimension.

        Returns:
            Basis matrix of shape (n, k).
        """
        X_scaled = (X - X.mean()) / (X.std() + 1e-8)
        B = np.column_stack([X_scaled ** i for i in range(1, k + 1)])
        return B

    def basis_matrix(self) -> np.ndarray:
        """Return basis matrix, shape (n, k)."""
        return self.B

    def penalty_matrix(self) -> np.ndarray:
        """Return identity penalty matrix, shape (k, k)."""
        return self.S

    def predict(self, x_new: np.ndarray | pd.Series) -> np.ndarray:
        """Build basis for new data.

        Args:
            x_new: New observations of the random effect variable.

        Returns:
            Basis matrix for new data.
        """
        if isinstance(x_new, pd.Series):
            x_vals = x_new.values
        else:
            x_vals = np.asarray(x_new)

        if self.is_factor:
            n_new = len(x_vals)
            k = self.k
            B_new = np.zeros((n_new, k))
            level_map = {lv: i for i, lv in enumerate(self.levels)}
            for i, val in enumerate(x_vals):
                idx = level_map.get(str(val))
                if idx is not None:
                    B_new[i, idx] = 1.0
            return B_new
        else:
            x_vals = x_vals.astype(float)
            k = self.k
            x_scaled = (x_vals - x_vals.mean()) / (x_vals.std() + 1e-8)
            return np.column_stack([x_scaled ** i for i in range(1, k + 1)])
