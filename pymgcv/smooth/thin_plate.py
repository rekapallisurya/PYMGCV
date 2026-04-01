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
        k: int | None = None,
        knot_indices: np.ndarray | None = None,
        m: int = 2,
        shrink: bool = False,
    ) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n, self.d = X.shape
        if self.n < 3:
            raise ValueError(f"Need at least 3 observations, got {self.n}")

        self.X = X
        self.m = m
        self.shrink = shrink
        self.M = self.d + 1  # null-space dimension for m=2
        self.k = k if k is not None else min(self.n, 10)

        if self.k > self.n:
            self.k = self.n
            warnings.warn(f"Basis dimension k reduced to n={self.n}.", UserWarning)
        if self.k < self.M + 1:
            self.k = self.M + 1
            warnings.warn(
                f"k increased to M+1={self.k} (minimum identifiable " f"dimension).",
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
                dists = spatial.distance.cdist(km.cluster_centers_, self.X)
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
        dists_kk = spatial.distance.cdist(self.knots, self.knots, "euclidean")
        E = _tps_kernel(dists_kk, self.d, self.m)
        E = 0.5 * (E + E.T)  # symmetrise

        # 2. Polynomial null-space matrix T  (nk x M)
        T = np.column_stack([np.ones(nk), self.knots])

        # 3. QR of T -> orthogonal complement Z
        Q_full = linalg.qr(T, mode="full")[0]  # (nk, nk)
        Z = Q_full[:, M:]  # (nk, nk - M)

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
            dists_xk = spatial.distance.cdist(self.X, self.knots, "euclidean")
            E_xk = _tps_kernel(dists_xk, self.d, self.m)
            T_x = np.column_stack([np.ones(self.n), self.X])
            B_null = T_x
            B_range = E_xk @ F_pred

        self.B = np.column_stack([B_null, B_range])

        # 5d. Penalty: identity on range, zero on null (after reparam)
        self.S = np.zeros((k, k))
        self.S[M:, M:] = np.eye(n_keep)

        # 5e. Shrinkage (bs='ts'): penalize null space too so smooth → 0
        if self.shrink:
            self.S[:M, :M] = np.eye(M)

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
        C = self.B.mean(axis=0).reshape(-1, 1)  # (k, 1)
        Q_con, _ = linalg.qr(C, mode="full")  # Q: (k, k)
        Z_con = Q_con[:, 1:]  # (k, k-1)

        self.B = self.B @ Z_con
        self.S = Z_con.T @ self.S @ Z_con
        self._constraint_Z = Z_con  # store for predict
        self._col_means = None  # not used with QR
        self.k = self.B.shape[1]

    # ------------------------------------------------------------------
    # RBF helper (legacy compatibility)
    # ------------------------------------------------------------------

    def _construct_rbf_matrix(
        self,
        X1: np.ndarray | None = None,
        X2: np.ndarray | None = None,
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
            raise ValueError(f"X_new has dim {X_new.shape[1]}, expected {self.d}")

        T_new = np.column_stack([np.ones(len(X_new)), X_new])
        dists_new = spatial.distance.cdist(X_new, self.knots, "euclidean")
        E_new = _tps_kernel(dists_new, self.d, self.m)
        B_range_new = E_new @ self._F_k
        B_raw = np.column_stack([T_new, B_range_new])

        # Apply same QR constraint rotation as training
        return B_raw @ self._constraint_Z

    # Legacy alias
    predict = predict_basis


def thin_plate_basis(
    X: np.ndarray,
    k: int | None = None,
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


# ---------------------------------------------------------------------------
# Duchon Spline (bs='ds') — generalized thin plate spline
# ---------------------------------------------------------------------------


def _duchon_kernel(r: np.ndarray, alpha: int) -> np.ndarray:
    """Evaluate Duchon spline radial kernel with given alpha = 2m - d + 2s."""
    eps = np.finfo(float).eps * 100
    out = np.zeros_like(r, dtype=np.float64)
    mask = r > eps
    if alpha % 2 == 0:
        out[mask] = r[mask] ** alpha * np.log(r[mask])
    else:
        out[mask] = r[mask] ** alpha
    return out


def _polynomial_basis(X: np.ndarray, max_degree: int) -> np.ndarray:
    """Multivariate polynomial basis up to given total degree.

    Parameters
    ----------
    X : array, shape (n, d)
    max_degree : int
        Maximum total polynomial degree. -1 returns empty matrix.

    Returns
    -------
    T : array, shape (n, M) where M = C(max_degree + d, d)
    """
    from itertools import combinations_with_replacement

    n, d = X.shape
    if max_degree < 0:
        return np.empty((n, 0))

    cols = []
    for deg in range(max_degree + 1):
        for powers in combinations_with_replacement(range(d), deg):
            col = np.ones(n)
            for p in powers:
                col = col * X[:, p]
            cols.append(col)
    return np.column_stack(cols) if cols else np.ones((n, 1))


class DuchonSpline:
    """Duchon spline basis (bs='ds') — generalized thin plate spline.

    Generalizes TPS by allowing different penalty orders and null space
    dimensions. The parameter s controls the null space: s=0 gives minimal
    null space, higher s gives larger null space.

    Parameters
    ----------
    X : array, shape (n, d)
        Input data.
    k : int, optional
        Basis dimension.
    m : int
        Derivative penalty order. Default 2.
    s : int
        Null space extra order. Default 0.
    shrink : bool
        If True, penalize null space too (shrinkage).

    References
    ----------
    Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms.
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int | None = None,
        m: int = 2,
        s: int = 0,
        shrink: bool = False,
    ) -> None:
        import math
        from math import comb

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n, self.d = X.shape
        self.X = X
        self.m = m
        self.s = s
        self.shrink = shrink

        # Duchon kernel exponent
        self.alpha = 2 * m - self.d + 2 * s
        if self.alpha <= 0:
            raise ValueError(
                f"Invalid Duchon parameters: 2m-d+2s = {self.alpha} must be > 0. "
                f"Got m={m}, d={self.d}, s={s}."
            )

        # Null space: polynomials of total degree <= ceil(m - d/2 + s) - 1
        null_degree = math.ceil(m - self.d / 2 + s) - 1
        self.null_degree = max(null_degree, 0)
        self.M = comb(self.null_degree + self.d, self.d)

        self.k = k if k is not None else min(self.n, 10)
        if self.k > self.n:
            self.k = self.n
        if self.M > 0 and self.k < self.M + 1:
            self.k = self.M + 1

        # Use all data as knots for small n
        if self.n <= 2000:
            self.knots = X.copy()
        else:
            idx = np.round(np.linspace(0, self.n - 1, min(self.k * 4, self.n))).astype(int)
            self.knots = X[idx]
        self.nk = len(self.knots)

        self.B: np.ndarray = np.zeros((self.n, self.k))
        self.S: np.ndarray = np.zeros((self.k, self.k))
        self._construct_basis()

    def _construct_basis(self) -> None:
        """Construct Duchon spline basis via eigendecomposition."""
        nk = self.nk
        M = self.M
        k = self.k

        # 1. Kernel matrix at knots
        dists_kk = spatial.distance.cdist(self.knots, self.knots, "euclidean")
        E = _duchon_kernel(dists_kk, self.alpha)
        E = 0.5 * (E + E.T)

        # 2. Polynomial null-space matrix
        T = _polynomial_basis(self.knots, self.null_degree)

        if M == 0:
            # No null space — use full eigen-decomposition of E
            eigvals, eigvecs = linalg.eigh(E)
            order = np.argsort(-np.abs(eigvals))
            n_keep = k
            D_k = eigvals[order[:n_keep]]
            U_k = eigvecs[:, order[:n_keep]]
            D_abs_safe = np.maximum(np.abs(D_k), 1e-10)
            reparam_scale = 1.0 / np.sqrt(D_abs_safe)
            U_k_reparam = U_k * reparam_scale[np.newaxis, :]
            F_pred = np.linalg.lstsq(E, U_k_reparam, rcond=None)[0]

            if nk == self.n:
                self.B = E @ F_pred
            else:
                dists_xk = spatial.distance.cdist(self.X, self.knots, "euclidean")
                E_xk = _duchon_kernel(dists_xk, self.alpha)
                self.B = E_xk @ F_pred

            self.S = np.eye(k)
        else:
            # 3. QR decompose T
            Q_full = linalg.qr(T, mode="full")[0]
            Z = Q_full[:, M:]

            # 4. Eigen-decompose Z'EZ
            ZtEZ = Z.T @ E @ Z
            ZtEZ = 0.5 * (ZtEZ + ZtEZ.T)
            eigvals, eigvecs = linalg.eigh(ZtEZ)
            order = np.argsort(-np.abs(eigvals))
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            n_keep = k - M
            D_k = eigvals[:n_keep]
            U_k = eigvecs[:, :n_keep]

            D_abs_safe = np.maximum(np.abs(D_k), 1e-10)
            reparam_scale = 1.0 / np.sqrt(D_abs_safe)
            U_k_reparam = U_k * reparam_scale[np.newaxis, :]

            F_pred = np.linalg.lstsq(E, Z @ U_k_reparam, rcond=None)[0]

            if nk == self.n:
                B_null = T.copy()
                B_range = E @ F_pred
            else:
                dists_xk = spatial.distance.cdist(self.X, self.knots, "euclidean")
                E_xk = _duchon_kernel(dists_xk, self.alpha)
                T_x = _polynomial_basis(self.X, self.null_degree)
                B_null = T_x
                B_range = E_xk @ F_pred

            self.B = np.column_stack([B_null, B_range])

            self.S = np.zeros((k, k))
            self.S[M:, M:] = np.eye(n_keep)
            if self.shrink:
                self.S[:M, :M] = np.eye(M)

        # QR constraint absorption
        C = self.B.mean(axis=0).reshape(-1, 1)
        Q_con, _ = linalg.qr(C, mode="full")
        Z_con = Q_con[:, 1:]
        self.B = self.B @ Z_con
        self.S = Z_con.T @ self.S @ Z_con
        self._constraint_Z = Z_con
        self.k = self.B.shape[1]

        self._T = T
        self._F_k = F_pred

    def basis_matrix(self) -> np.ndarray:
        return self.B

    def penalty_matrix_S(self) -> np.ndarray:
        return self.S

    def predict_basis(self, X_new: np.ndarray) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=np.float64)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        T_new = _polynomial_basis(X_new, self.null_degree)
        dists_new = spatial.distance.cdist(X_new, self.knots, "euclidean")
        E_new = _duchon_kernel(dists_new, self.alpha)
        if self.M > 0:
            B_raw = np.column_stack([T_new, E_new @ self._F_k])
        else:
            B_raw = E_new @ self._F_k
        return B_raw @ self._constraint_Z

    predict = predict_basis


# ---------------------------------------------------------------------------
# Spherical Spline (bs='sos') — thin plate spline on a sphere
# ---------------------------------------------------------------------------


class SphericalSpline:
    """Spline on the sphere (bs='sos').

    Two-dimensional spline on a sphere using great-circle distances.
    Input variables are latitude and longitude (in degrees).

    Parameters
    ----------
    lat : array, shape (n,)
        Latitude in degrees.
    lon : array, shape (n,)
        Longitude in degrees.
    k : int, optional
        Basis dimension.

    References
    ----------
    Wahba, G. (1981). Spline interpolation and smoothing on the sphere.
    """

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        k: int | None = None,
    ) -> None:
        lat = np.asarray(lat, dtype=np.float64).ravel()
        lon = np.asarray(lon, dtype=np.float64).ravel()
        if len(lat) != len(lon):
            raise ValueError("lat and lon must have the same length")

        self.n = len(lat)
        self.lat_deg = lat
        self.lon_deg = lon
        self.lat = np.deg2rad(lat)
        self.lon = np.deg2rad(lon)
        self.k = k if k is not None else min(self.n, 10)

        # Null space = constant (on sphere surface)
        self.M = 1
        if self.k < self.M + 1:
            self.k = self.M + 1

        self._construct_basis()

    @staticmethod
    def _great_circle_dist(lat1, lon1, lat2, lon2):
        """Great circle distance matrix using Haversine formula."""
        dlat = lat1[:, None] - lat2[None, :]
        dlon = lon1[:, None] - lon2[None, :]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1[:, None]) * np.cos(lat2[None, :]) * np.sin(dlon / 2) ** 2
        )
        a = np.clip(a, 0, 1)
        return 2 * np.arcsin(np.sqrt(a))

    def _construct_basis(self) -> None:
        """Construct spherical spline basis."""
        k = self.k
        M = self.M
        n = self.n

        # Great-circle distance matrix
        D = self._great_circle_dist(self.lat, self.lon, self.lat, self.lon)

        # TPS kernel on sphere (d=2, m=2): r^2 * log(r)
        E = _tps_kernel(D, d=2, m=2)
        E = 0.5 * (E + E.T)

        # Null space: constant
        T = np.ones((n, 1))

        # QR of T
        Q_full = linalg.qr(T, mode="full")[0]
        Z = Q_full[:, M:]

        # Eigen-decompose Z'EZ
        ZtEZ = Z.T @ E @ Z
        ZtEZ = 0.5 * (ZtEZ + ZtEZ.T)
        eigvals, eigvecs = linalg.eigh(ZtEZ)
        order = np.argsort(-np.abs(eigvals))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        n_keep = k - M
        D_k = eigvals[:n_keep]
        U_k = eigvecs[:, :n_keep]

        D_abs_safe = np.maximum(np.abs(D_k), 1e-10)
        reparam_scale = 1.0 / np.sqrt(D_abs_safe)
        U_k_reparam = U_k * reparam_scale[np.newaxis, :]

        F_pred = np.linalg.lstsq(E, Z @ U_k_reparam, rcond=None)[0]

        B_null = T.copy()
        B_range = E @ F_pred
        self.B = np.column_stack([B_null, B_range])

        self.S = np.zeros((k, k))
        self.S[M:, M:] = np.eye(n_keep)

        # QR constraint absorption
        C = self.B.mean(axis=0).reshape(-1, 1)
        Q_con, _ = linalg.qr(C, mode="full")
        Z_con = Q_con[:, 1:]
        self.B = self.B @ Z_con
        self.S = Z_con.T @ self.S @ Z_con
        self._constraint_Z = Z_con
        self.k = self.B.shape[1]

        self._F_k = F_pred

    def basis_matrix(self) -> np.ndarray:
        return self.B

    def penalty_matrix_S(self) -> np.ndarray:
        return self.S

    def predict_basis(self, lat_new: np.ndarray, lon_new: np.ndarray) -> np.ndarray:
        lat_new = np.deg2rad(np.asarray(lat_new, dtype=np.float64).ravel())
        lon_new = np.deg2rad(np.asarray(lon_new, dtype=np.float64).ravel())
        D_new = self._great_circle_dist(lat_new, lon_new, self.lat, self.lon)
        E_new = _tps_kernel(D_new, d=2, m=2)
        T_new = np.ones((len(lat_new), 1))
        B_raw = np.column_stack([T_new, E_new @ self._F_k])
        return B_raw @ self._constraint_Z

    predict = predict_basis
