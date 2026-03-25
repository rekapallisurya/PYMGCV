"""Thin plate regression spline basis construction.

Implements exact thin plate regression splines (TPRS) following the mgcv
algorithm (Wood 2003).  The construction uses the eigen-decomposition of
the full-knot-set penalty matrix to produce an orthogonally reparameterized
rank-k basis with a diagonal penalty -- exactly matching mgcv.

Algorithm (Wood 2003, section 3.1):
  1. Build the full TPS penalty matrix E from the RBF kernel evaluated at
     *all n* data points (or a sub-sample used as knots).
  2. Build the polynomial null-space matrix T.
  3. Absorb the null space via QR:  T = Q R  then  Z = Q_perp  (columns
     orthogonal to the null space).
  4. Eigen-decompose Z'EZ = U D U' and keep the top (k - M) eigenvectors.
  5. The basis is B = [T | Z U_k],  penalty S is block-diag(0_M, D_k).

References:
    Wood, S. N. (2003). Thin plate regression splines. JRSS(B), 65(1).
    Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms.

Module exports:
    ThinPlateSpline: Main TPRS basis class
    thin_plate_basis: Functional API for basis matrix construction
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy import linalg, spatial


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _tps_kernel(r: np.ndarray, d: int, m: int = 2) -> np.ndarray:
    r"""Evaluate the thin-plate-spline radial kernel.

    For d=1, m=2:  phi(r) = r^3  (odd: 2m-d = 3)
    For d=2, m=2:  phi(r) = r^2 log(r)  (even: 2m-d = 2)

    Unnormalised form consistent with mgcv internal representation.
    """
    alpha = 2 * m - d
    eps = np.finfo(float).eps * 100
    out = np.zeros_like(r, dtype=np.float64)
    mask = r > eps
    if alpha % 2 == 0:
        out[mask] = r[mask] ** alpha * np.log(r[mask])
    else:
        out[mask] = r[mask] ** alpha
    return out


class ThinPlateSpline:
    """Thin plate regression spline basis (mgcv-equivalent).

    Constructs a TPRS basis matrix B of shape (n, k) and a matching
    penalty matrix S of shape (k, k) via the eigendecomposition
    procedure described in Wood (2003).

    Parameters
    ----------
    X : array, shape (n, d)
        Input data.
    k : int, optional
        Basis dimension. Default: min(n, 40).
    knot_indices : array of int, optional
        Pre-selected knot indices into X.
    m : int
        Derivative penalty order. Default: 2.

    Attributes
    ----------
    B : array, shape (n, k)
        Basis matrix evaluated at input data.
    S : array, shape (k, k)
        Penalty matrix with block-diag(0_M, diag(|D_k|)) structure.
    knots : array, shape (nk, d)
        Knot locations.
    """

    def __init__(
        self,
        X: np.ndarray,
        k: Optional[int] = None,
        knot_indices: Optional[np.ndarray] = None,
        m: int = 2,
    ) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n, self.d = X.shape
        if self.n < 3:
            raise ValueError(f"Need at least 3 observations, got {self.n}")

        self.X = X
        self.m = m
        self.M = self.d + 1               # null-space dimension for m=2
        self.k = k if k is not None else min(self.n, 10)

        if self.k > self.n:
            self.k = self.n
            warnings.warn(
                f"Basis dimension k reduced to n={self.n}.", UserWarning
            )
        if self.k < self.M + 1:
            self.k = self.M + 1
            warnings.warn(
                f"k increased to M+1={self.k} (minimum identifiable "
                f"dimension).",
                UserWarning,
            )

        # Select knots
        if knot_indices is not None:
            self.knot_indices = np.asarray(knot_indices)
        else:
            self.knot_indices = self._select_knots()
        self.knots = X[self.knot_indices]
        self.nk = len(self.knots)

        # Construct basis & penalty via eigendecomposition
        self.B: np.ndarray = np.zeros((self.n, self.k))
        self.S: np.ndarray = np.zeros((self.k, self.k))
        self._construct_basis()

    # ------------------------------------------------------------------
    # Knot selection
    # ------------------------------------------------------------------

    def _select_knots(self) -> np.ndarray:
        """Return indices of knots.

        For n <= 2000 use every data point (matching mgcv default) so
        the penalty matrix is the *exact* TPS penalty.  For larger n
        sub-sample via quantiles (1-d) or k-means (multi-d).
        """
        if self.n <= 2000:
            return np.arange(self.n)

        nk = min(max(self.k * 4, 200), self.n)
        if self.d == 1:
            qs = np.linspace(0, self.n - 1, nk, dtype=int)
            return np.argsort(self.X[:, 0])[qs]
        else:
            try:
                from sklearn.cluster import KMeans

                km = KMeans(n_clusters=nk, random_state=42, n_init=10)
                km.fit(self.X)
                dists = spatial.distance.cdist(
                    km.cluster_centers_, self.X
                )
                return np.argmin(dists, axis=1)
            except ImportError:
                idx = np.arange(self.n)
                rng = np.random.RandomState(42)
                rng.shuffle(idx)
                return idx[:nk]

    # ------------------------------------------------------------------
    # Core: eigendecomposition-based basis (Wood 2003 s3)
    # ------------------------------------------------------------------

    def _construct_basis(self) -> None:
        r"""Construct the TPRS basis B and penalty S.

        Steps (Wood 2003):
        1. Evaluate the TPS kernel at all knot pairs -> E (nk x nk).
        2. Build polynomial null-space matrix T (nk x M).
        3. QR-decompose T to get the orthogonal complement Z.
        4. Eigen-decompose Z'EZ; keep top (k - M) eigenvectors.
        5. Assemble B and S.
        """
        nk = self.nk
        M = self.M
        k = self.k

        # 1. Full kernel matrix at knots
        dists_kk = spatial.distance.cdist(
            self.knots, self.knots, "euclidean"
        )
        E = _tps_kernel(dists_kk, self.d, self.m)
        E = 0.5 * (E + E.T)  # symmetrise

        # 2. Polynomial null-space matrix T  (nk x M)
        T = np.column_stack([np.ones(nk), self.knots])

        # 3. QR of T -> orthogonal complement Z
        Q_full = linalg.qr(T, mode="full")[0]  # (nk, nk)
        Z = Q_full[:, M:]                       # (nk, nk - M)

        # 4. Eigen-decompose Z'EZ
        ZtEZ = Z.T @ E @ Z
        ZtEZ = 0.5 * (ZtEZ + ZtEZ.T)
        eigvals, eigvecs = linalg.eigh(ZtEZ)

        # Sort descending by absolute eigenvalue
        order = np.argsort(-np.abs(eigvals))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Keep top (k - M) components
        n_keep = k - M
        D_k = eigvals[:n_keep]
        U_k = eigvecs[:, :n_keep]

        # 5a. Penalty-absorbing reparameterization (mgcv default).
        # Divide range eigenvectors by sqrt(|eigenvalue|) so the penalty
        # becomes identity — all wiggly components penalised equally.
        D_abs_safe = np.maximum(np.abs(D_k), 1e-10)
        reparam_scale = 1.0 / np.sqrt(D_abs_safe)
        U_k_reparam = U_k * reparam_scale[np.newaxis, :]

        # 5b. Prediction transform (with reparameterised eigenvectors)
        F_pred = np.linalg.lstsq(E, Z @ U_k_reparam, rcond=None)[0]

        # 5c. Build basis evaluated at all n data points
        if nk == self.n:
            B_null = T.copy()
            B_range = E @ F_pred  # = Z @ U_k_reparam
        else:
            dists_xk = spatial.distance.cdist(
                self.X, self.knots, "euclidean"
            )
            E_xk = _tps_kernel(dists_xk, self.d, self.m)
            T_x = np.column_stack([np.ones(self.n), self.X])
            B_null = T_x
            B_range = E_xk @ F_pred

        self.B = np.column_stack([B_null, B_range])

        # 5d. Penalty: identity on range, zero on null (after reparam)
        self.S = np.zeros((k, k))
        self.S[M:, M:] = np.eye(n_keep)

        # 6. QR constraint absorption (mgcv's absorb.cons)
        self._apply_sum_to_zero_constraint()

        # Store internals for prediction
        self._T_knots = T
        self._F_k = F_pred
        self._D_k = D_k
        self._U_k = U_k
        self._Z = Z

    def _apply_sum_to_zero_constraint(self) -> None:
        """Absorb sum-to-zero constraint via QR (mgcv's absorb.cons).

        The constraint 1'B\u03b2 = 0 (mean of smooth is zero) is absorbed
        into the basis by rotating the parameter space so the constrained
        direction is dropped.  Both B and S are transformed.
        """
        C = self.B.mean(axis=0).reshape(-1, 1)       # (k, 1)
        Q_con, _ = linalg.qr(C, mode='full')         # Q: (k, k)
        Z_con = Q_con[:, 1:]                          # (k, k-1)

        self.B = self.B @ Z_con
        self.S = Z_con.T @ self.S @ Z_con
        self._constraint_Z = Z_con                    # store for predict
        self._col_means = None                        # not used with QR
        self.k = self.B.shape[1]

    # ------------------------------------------------------------------
    # RBF helper (legacy compatibility)
    # ------------------------------------------------------------------

    def _construct_rbf_matrix(
        self,
        X1: Optional[np.ndarray] = None,
        X2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate TPS kernel between two point sets."""
        if X1 is None:
            X1 = self.X
        if X2 is None:
            X2 = self.knots
        dists = spatial.distance.cdist(X1, X2, "euclidean")
        return _tps_kernel(dists, self.d, self.m)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def basis_matrix(self) -> np.ndarray:
        """Return the basis matrix B, shape (n, k)."""
        return self.B

    def penalty_matrix_S(self) -> np.ndarray:
        """Return the penalty matrix S, shape (k, k)."""
        return self.S

    def predict_basis(self, X_new: np.ndarray) -> np.ndarray:
        """Evaluate basis at new points.

        Parameters
        ----------
        X_new : array, shape (n_new, d)
            New input points.

        Returns
        -------
        B_new : array, shape (n_new, k)
            Basis matrix at new points.
        """
        X_new = np.asarray(X_new, dtype=np.float64)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        if X_new.shape[1] != self.d:
            raise ValueError(
                f"X_new has dim {X_new.shape[1]}, expected {self.d}"
            )

        T_new = np.column_stack([np.ones(len(X_new)), X_new])
        dists_new = spatial.distance.cdist(
            X_new, self.knots, "euclidean"
        )
        E_new = _tps_kernel(dists_new, self.d, self.m)
        B_range_new = E_new @ self._F_k
        B_raw = np.column_stack([T_new, B_range_new])

        # Apply same QR constraint rotation as training
        return B_raw @ self._constraint_Z

    # Legacy alias
    predict = predict_basis


def thin_plate_basis(
    X: np.ndarray,
    k: Optional[int] = None,
) -> np.ndarray:
    """Functional API for thin plate regression spline basis.

    Parameters
    ----------
    X : array, shape (n, d)
        Input data.
    k : int, optional
        Basis dimension. Default: min(n, 40).

    Returns
    -------
    B : array, shape (n, k)
        Basis matrix.
    """
    tprs = ThinPlateSpline(X, k=k)
    return tprs.basis_matrix()
