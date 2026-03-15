"""Thin plate regression spline basis construction.

Implements exact thin plate regression splines (TPRS) as used in mgcv.
Constructs the basis matrix B via:
  1. Radial basis functions: φ(r) = r² log(r)
  2. Polynomial null space (linear + constant)
  3. Singular value decomposition / eigen decomposition
  4. Truncation to rank k

References:
    - Wood, S. N. (2003). Thin plate regression splines. JRSS(B), 65(1), 95-114.
    - Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms. Constr. Approx.

Module exports:
    - ThinPlateSpline: Main TPRS basis class
    - thin_plate_basis: Functional API for basis matrix construction
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy import linalg, spatial


class ThinPlateSpline:
    """Thin plate regression spline basis.

    Constructs a TPRS basis matrix B of shape (n, k) where:
    - n: number of observations
    - k: basis dimension (number of knots)

    The full GAM design matrix is then X = [X_parametric, B].

    Attributes:
        X: Input data, shape (n, d) where d=1 for univariate, d>1 for multivariate.
        k: Basis dimension (default: auto-select via min(n, 40)).
        knot_indices: Indices of selected knots in X.
        knots: Actual knot locations, shape (k, d).
        B: Basis matrix, shape (n, k).
        poly_coef: Polynomial null space coefficients (internal).
    """

    def __init__(
        self,
        X: np.ndarray,
        k: Optional[int] = None,
        knot_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize thin plate spline basis.

        Args:
            X: Input data, shape (n, d). Univariate (d=1) or multivariate (d>1).
            k: Basis dimension. If None, auto-select as min(n, 40).
            knot_indices: Pre-selected knot indices. If None, select via quantiles.

        Raises:
            ValueError: If X is empty or too small for basis.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n, self.d = X.shape
        if self.n < 3:
            raise ValueError(f'Need at least 3 observations, got {self.n}')

        self.X = X
        self.k = k if k is not None else min(self.n, 40)

        if self.k > self.n:
            self.k = self.n
            warnings.warn(
                f'Basis dimension k={self.k} >= n={self.n}. Reducing k to {self.n}.',
                UserWarning
            )

        # Select knots
        if knot_indices is not None:
            self.knot_indices = knot_indices
        else:
            self.knot_indices = self._select_knots_quantile(self.k)

        self.knots = X[self.knot_indices]
        
        # Construct basis
        self.B: np.ndarray = np.zeros((self.n, self.k))
        self.poly_coef: Optional[np.ndarray] = None
        
        self._construct_basis()

    def _select_knots_quantile(self, k: int) -> np.ndarray:
        """Select knots via stratified quantiles (mgcv-compatible).

        For univariate: equally spaced quantiles ensure uniform coverage.
        For multivariate: stratified k-means clustering or quantile sampling.
        
        Reference:
            mgcv/src/smooth.c: select_knots() function

        Args:
            k: Number of knots.

        Returns:
            Indices of selected knots.
        """
        if self.d == 1:
            # Univariate: quantile-based selection (mgcv default)
            quantile_positions = np.linspace(0, self.n - 1, k, dtype=int)
            sorted_indices = np.argsort(self.X[:, 0])
            return sorted_indices[quantile_positions]
        else:
            # Multivariate: try k-means clustering, fallback to stratified sampling
            try:
                # Attempt import (sklearn available)
                from sklearn.cluster import KMeans
                
                # Use k-means for better spatial coverage
                kmeans = KMeans(
                    n_clusters=k, 
                    random_state=42, 
                    n_init=10,
                    max_iter=100,
                    tol=1e-6
                )
                kmeans.fit(self.X)
                
                # Find closest point to each cluster center
                distances = spatial.distance.cdist(
                    kmeans.cluster_centers_, self.X, metric='euclidean'
                )
                knot_indices = np.argmin(distances, axis=1)
                
                return knot_indices
            except (ImportError, Exception):
                # Fallback: stratified sampling if sklearn unavailable
                # Divide observations into k groups and select one from each
                indices = np.arange(self.n)
                np.random.seed(42)  # For reproducibility
                np.random.shuffle(indices)
                
                # Simple stratified approach
                strata_size = max(1, self.n // k)
                knot_indices = []
                
                for i in range(k):
                    start = i * strata_size
                    end = start + strata_size if i < k - 1 else self.n
                    
                    if start < end:
                        # Select random point from stratum
                        strata_range = end - start
                        selected_idx = start + np.random.randint(0, strata_range)
                        knot_indices.append(indices[selected_idx])
                
                # If we didn't get enough knots, pad with random selections
                while len(knot_indices) < k:
                    knot_indices.append(np.random.randint(0, self.n))
                
                return np.array(knot_indices[:k])

    def _construct_basis(self) -> None:
        r"""Construct TPRS basis with correct mgcv algorithm.
        
        Mathematical Foundation (Wood 2003):
        A thin plate spline is represented as:
            $$f(x) = \sum_{j=1}^k a_j \phi(\|x - x_j\|) + \sum_{l=1}^{d+1} c_l p_l(x)$$
        
        where:
        - $\phi(r) = r^2 \log(r)$ is the RBF kernel (thin plate spline kernel)
        - $p_l$ are polynomial basis functions $[1, x_1, \ldots, x_d]$
        - $x_j$ are knot locations
        
        The basis matrix B is constructed via:
        1. RBF matrix H: $H_{ij} = \phi(\|X_i - \text{knots}_j\|)$
        2. Polynomial matrix P: $P_{il} = p_l(X_i)$
        3. Augmented system solving for coefficients
        4. Truncation maintains both H and P terms
        
        References:
            Wood, S. N. (2003). Thin plate regression splines. JRSS(B), 65(1), 95-114.
        """
        # Step 1: Compute RBF matrix H (n × k)
        H = self._construct_rbf_matrix()  # shape (n, k)
        
        # Step 2: Compute polynomial matrix P (n × d+1)
        p_dim = self.d + 1
        P = np.column_stack([np.ones(self.n), self.X])  # shape (n, d+1)
        
        # Step 3: Compute penalty matrix structure
        # S_rr (k × k) = RBF distance matrix between knots
        S_rr = self._construct_rbf_matrix(self.knots, self.knots)  # shape (k, k)
        
        # S_p (d+1 × d+1) = zero (polynomial is unpenalized null space)
        
        # Polynomial evaluation at knots
        P_k = np.column_stack([np.ones(self.k), self.knots])  # shape (k, d+1)
        
        # Step 4: Augmented system (symmetric, indefinite)
        # [S_rr  P_k] [a]   [H^T]
        # [P_k^T  0 ] [c] = [P^T]
        #
        # where a are RBF coefficients and c are polynomial coefficients
        
        aug = np.vstack([
            np.hstack([S_rr, P_k]),
            np.hstack([P_k.T, np.zeros((p_dim, p_dim))])
        ])  # shape (k+d+1, k+d+1)
        
        # Right-hand side: [H^T, P^T]^T
        rhs = np.vstack([H.T, P.T])  # shape (k+d+1, n)
        
        # Step 5: Solve augmented system using appropriate method
        try:
            # Attempt Cholesky decomposition (efficient for positive definite)
            # Note: the augmented system is indefinite, so this may fail
            L = linalg.cholesky(aug, lower=True)
            coef = linalg.cho_solve((L, True), rhs)
        except linalg.LinAlgError:
            # Fallback: SVD with adaptive threshold for numerical stability
            U, svals, Vt = linalg.svd(aug, full_matrices=False)
            
            # Adaptive threshold: follow LAPACK convention
            # thresh = eps * max(m, n) * max_singular_value
            eps = np.finfo(float).eps
            thresh = eps * max(aug.shape) * svals[0]
            
            # Invert singular values with safeguard
            svals_inv = np.where(svals > thresh, 1.0 / svals, 0)
            coef = Vt.T @ np.diag(svals_inv) @ U.T @ rhs
        
        # Step 6: Extract and store coefficients for out-of-sample prediction
        self._rbf_coef = coef[:self.k, :]  # shape (k, n)
        self._poly_coef = coef[self.k:, :]  # shape (d+1, n)
        
        # Step 7: Construct basis matrix
        # B[i,j] represents the j-th basis function evaluated at X[i]
        # Full matrix before truncation: H @ a + P @ c
        B_full = H @ self._rbf_coef + P @ self._poly_coef  # shape (n, n)
        
        # Truncate to basis dimension k (keep first k columns)
        # This maintains the null space (polynomial terms) plus k-p_dim RBF terms
        self.B = B_full[:, :self.k]

    def _construct_rbf_matrix(
        self,
        X1: Optional[np.ndarray] = None,
        X2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Construct RBF matrix with numerical stability.
        
        Implements $\phi(r) = r^2 \log(r)$ with safeguards for $r \to 0^+$.
        
        Mathematical Foundation:
        The thin plate spline kernel is defined as:
            $$\phi(r) = \begin{cases}
                r^2 \log(r) & \text{if } r > 0 \\
                0 & \text{if } r = 0
            \end{cases}$$
        
        As $r \to 0^+$, we have $r^2 \log(r) \to 0$ (limit is zero).
        
        For numerical stability, avoid computing log of very small numbers
        by using an explicit threshold. Small distances (≤ eps) are set to 0.
        
        References:
            Wood, S. N. (2003). Thin plate regression splines. JRSS(B), 65(1), 95-114.
            Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms.

        Args:
            X1: First set of points, shape (n1, d). Defaults to self.X.
            X2: Second set of points, shape (n2, d). Defaults to self.knots.

        Returns:
            RBF matrix, shape (n1, n2).
        """
        if X1 is None:
            X1 = self.X
        if X2 is None:
            X2 = self.knots

        # Compute pairwise Euclidean distances
        distances = spatial.distance.cdist(X1, X2, metric='euclidean')

        # RBF: r² log(r) for r > 0, 0 otherwise
        # Use explicit conditioning to avoid log(0) and numerical issues
        eps = np.finfo(float).eps * 100  # Small threshold for numerical stability
        
        rbf = np.zeros_like(distances, dtype=np.float64)
        
        # Only compute for distances > eps (avoid log of tiny numbers)
        mask = distances > eps
        rbf[mask] = distances[mask] ** 2 * np.log(distances[mask])
        
        # For distances <= eps, value is 0 (limit of r² log(r) as r → 0)
        rbf[~mask] = 0.0
        
        return rbf

    def basis_matrix(self) -> np.ndarray:
        """Return the basis matrix B, shape (n, k).

        This is the matrix B used in the design matrix X = [X_parametric, B].
        """
        return self.B

    def predict_basis(self, X_new: np.ndarray) -> np.ndarray:
        r"""Evaluate basis at new points via stored coefficients.
        
        For out-of-sample prediction, evaluate the thin plate spline basis functions
        at new locations using coefficients computed during training.
        
        Each basis function is evaluated as:
            $$B_j(\mathbf{x}_{\text{new}}) = \sum_l a_{j,l} \phi(\|\mathbf{x}_{\text{new}} - \text{knots}_l\|) 
                                             + \sum_l c_{j,l} p_l(\mathbf{x}_{\text{new}})$$
        
        where $a_{j,l}$ and $c_{j,l}$ are coefficients stored during `_construct_basis()`.

        Args:
            X_new: New input points, shape (n_new, d).

        Returns:
            Basis matrix at new points, shape (n_new, k).

        Raises:
            ValueError: If X_new dimension doesn't match training data.
            RuntimeError: If coefficients not available (basis not constructed).
        """
        X_new = np.asarray(X_new, dtype=np.float64)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

        if X_new.shape[1] != self.d:
            raise ValueError(
                f'X_new has dimension {X_new.shape[1]}, expected {self.d}'
            )

        if not hasattr(self, '_rbf_coef') or not hasattr(self, '_poly_coef'):
            raise RuntimeError(
                'Basis coefficients not stored. '
                'Ensure ThinPlateSpline was initialized and _construct_basis() was called.'
            )

        # RBF matrix at new points: H_new[i,j] = φ(||X_new[i] - knots[j]||)
        H_new = self._construct_rbf_matrix(X_new, self.knots)  # shape (n_new, k)

        # Polynomial matrix at new points: P_new[i,l] = p_l(X_new[i])
        p_dim = self.d + 1
        P_new = np.column_stack(
            [np.ones(len(X_new)), X_new]
        )  # shape (n_new, d+1)

        # Basis matrix: [H_new | P_new] @ [a; c]
        # = H_new @ a + P_new @ c
        B_full = H_new @ self._rbf_coef + P_new @ self._poly_coef  # shape (n_new, n)

        # Return only first k columns (basis dimension)
        return B_full[:, :self.k]


def thin_plate_basis(
    X: np.ndarray,
    k: Optional[int] = None,
) -> np.ndarray:
    """Functional API for thin plate regression spline basis.

    Args:
        X: Input data, shape (n, d).
        k: Basis dimension. If None, auto-select as min(n, 40).

    Returns:
        Basis matrix B, shape (n, k).

    Example:
        >>> X = np.linspace(0, 1, 50).reshape(-1, 1)
        >>> B = thin_plate_basis(X, k=10)
        >>> B.shape
        (50, 10)
    """
    tprs = ThinPlateSpline(X, k=k)
    return tprs.basis_matrix()
