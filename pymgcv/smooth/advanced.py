"""Advanced smooth basis types.

Provides:
  - AdaptiveSpline     (bs='ad')  — spatially-varying smoothness
  - GPSmooth           (bs='gp')  — Gaussian process smooth
  - FactorSmooth       (bs='fs')  — grouped factor-level smooths
  - FactorDeviation    (bs='sz')  — factor smooth deviations

References:
    - Wood, S.N. (2017). GAMs: An Introduction with R.
    - Crainiceanu, C. et al. (2005). Exact likelihood ratio tests.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Adaptive Smooth (bs='ad')
# ---------------------------------------------------------------------------


class AdaptiveSpline:
    """Adaptive smooth with spatially-varying penalty (bs='ad').

    Uses P-spline basis with a spatial penalty that varies along x.
    The smoothness is allowed to vary over the range of x by using
    a second-level penalty on the log smoothing parameters.

    Implementation follows Wood (2017) §5.3.3.

    Attributes:
        X: Input data, shape (n,).
        k: Number of main B-spline coefficients.
        m: Difference order for penalty (default 2).
        B: Basis matrix, shape (n, k).
        S: List of component penalty matrices, one per adaptive level.
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int = 20,
        m: int = 2,
        n_adaptive: int = 5,
    ) -> None:
        from pymgcv.smooth.bspline import PSplineBasis, _diff_matrix

        self.X = np.asarray(X, dtype=float).ravel()
        self.n = len(self.X)
        self.k = k
        self.m = m
        self.n_adaptive = n_adaptive

        ps = PSplineBasis(self.X, k=k, m=m)
        self.B = ps.B
        self.knots = ps.knots

        # Build spatially-varying penalty blocks
        # Split coefficients into n_adaptive groups; each group has its own λ
        group_size = max(1, k // n_adaptive)
        D = _diff_matrix(k, m)  # (k-m, k)
        DtD = D.T @ D

        # Create a block-diagonal weighting matrix for each adaptive group
        self.S_list = []
        for g in range(n_adaptive):
            start = g * group_size
            end = min(start + group_size, k)
            W = np.zeros((k, k))
            W[start:end, start:end] = np.eye(end - start)
            # Penalty = DtD weighted by W
            self.S_list.append(DtD @ W @ DtD)

        # Combined S for single-lambda use
        self.S = sum(self.S_list)

    def basis_matrix(self) -> np.ndarray:
        return self.B

    def penalty_matrices(self) -> list[np.ndarray]:
        return self.S_list

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        from pymgcv.smooth.bspline import _bspline_design_matrix

        X_new = np.asarray(X_new, dtype=float).ravel()
        return _bspline_design_matrix(X_new, self.knots, 3)


# ---------------------------------------------------------------------------
# Gaussian Process Smooth (bs='gp')
# ---------------------------------------------------------------------------

_GP_KERNELS = {
    "exp_quad": lambda d, l: np.exp(-0.5 * (d / l) ** 2),  # squared-exponential
    "matern12": lambda d, l: np.exp(-d / l),  # Matérn 1/2
    "matern32": lambda d, l: (1 + np.sqrt(3) * d / l) * np.exp(-np.sqrt(3) * d / l),
    "matern52": lambda d, l: (1 + np.sqrt(5) * d / l + 5 * d**2 / (3 * l**2))
    * np.exp(-np.sqrt(5) * d / l),
    "rational_quadratic": lambda d, l: (1 + d**2 / (2 * l**2)) ** -1,
}


class GPSmooth:
    """Gaussian process smooth (bs='gp').

    The smooth is represented using a low-rank GP basis constructed from
    the spectral decomposition of the kernel matrix evaluated at a set of
    inducing points (knots).

    Attributes:
        X: Input data shape (n,) or (n, d).
        k: Number of basis functions (inducing points).
        kernel: Kernel name (default 'exp_quad').
        length_scale: GP length-scale parameter.
        B: Basis matrix, shape (n, k).
        S: Penalty matrix (inverse of kernel matrix at knots), shape (k, k).
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int = 10,
        kernel: str = "exp_quad",
        length_scale: float | None = None,
    ) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X = X
        self.n, self.d = X.shape
        self.k = min(k, self.n)
        self.kernel_name = kernel
        self._kern_fn = _GP_KERNELS.get(kernel, _GP_KERNELS["exp_quad"])

        if length_scale is None:
            # Heuristic: median inter-point distance
            dists = np.sqrt(
                np.sum(
                    (X[:: max(1, len(X) // 500)] - X[:: max(1, len(X) // 500), :].mean(0)) ** 2,
                    axis=1,
                )
            )
            length_scale = float(np.median(dists[dists > 0]) or 1.0)
        self.length_scale = length_scale

        # Select inducing points at quantiles / k-means
        self._select_inducing_points()
        self._build_basis()

    def _select_inducing_points(self) -> None:
        from scipy.cluster.vq import kmeans2

        if self.k >= self.n:
            self.inducing = self.X.copy()
        else:
            try:
                centroids, _ = kmeans2(self.X, self.k, minit="points", niter=10)
                self.inducing = centroids
            except Exception:
                idx = np.round(np.linspace(0, self.n - 1, self.k)).astype(int)
                self.inducing = self.X[idx]

    def _dist_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Euclidean distance matrix between rows of A and B."""
        diff = A[:, None, :] - B[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def _build_basis(self) -> None:
        l = self.length_scale
        # K_nm: cross-kernel between data and inducing points
        D_nm = self._dist_matrix(self.X, self.inducing)
        K_nm = self._kern_fn(D_nm, l)

        # K_mm: kernel at inducing points
        D_mm = self._dist_matrix(self.inducing, self.inducing)
        K_mm = self._kern_fn(D_mm, l)
        K_mm += 1e-6 * np.eye(self.k)  # nugget for stability

        # Nystrom approximation: B = K_nm @ K_mm^{-1/2}
        try:
            L = np.linalg.cholesky(K_mm)
            L_inv = np.linalg.inv(L)
            self.B = K_nm @ L_inv.T
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(K_mm)
            vals = np.maximum(vals, 1e-10)
            self.B = K_nm @ (vecs / np.sqrt(vals))

        # Penalty: S = K_mm^{-1} = L_inv.T @ L_inv
        try:
            L_inv = np.linalg.inv(np.linalg.cholesky(K_mm))
            self.S = L_inv.T @ L_inv
        except np.linalg.LinAlgError:
            self.S = np.linalg.pinv(K_mm)

    def basis_matrix(self) -> np.ndarray:
        return self.B

    def penalty_matrix(self) -> np.ndarray:
        return self.S

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        l = self.length_scale
        D = self._dist_matrix(X_new, self.inducing)
        K = self._kern_fn(D, l)
        D_mm = self._dist_matrix(self.inducing, self.inducing)
        K_mm = self._kern_fn(D_mm, l) + 1e-6 * np.eye(self.k)
        try:
            L = np.linalg.cholesky(K_mm)
            return K @ np.linalg.inv(L).T
        except np.linalg.LinAlgError:
            return K @ np.linalg.pinv(K_mm)


# ---------------------------------------------------------------------------
# Factor Smooth Interactions (bs='fs')
# ---------------------------------------------------------------------------


class FactorSmooth:
    """Factor smooth interaction basis (bs='fs').

    Creates one smooth per factor level, sharing the same smoothing parameter.
    This is more efficient than separate by-variable models because all levels
    share a single λ.

    The basis for level l is: B_l(x) * I(group == l), then all concatenated.

    Attributes:
        X: Continuous covariate, shape (n,).
        group: Factor variable, shape (n,).
        levels: Sorted list of factor levels.
        k: Number of basis functions per level.
        B: Combined basis matrix, shape (n, k * n_levels).
        S: Block penalty matrix, shape (k*n_levels, k*n_levels).
    """

    def __init__(
        self,
        X: np.ndarray,
        group: np.ndarray,
        k: int = 10,
        basis: str = "cr",
    ) -> None:
        from pymgcv.smooth.bspline import PSplineBasis
        from pymgcv.smooth.cubic_spline import CubicRegressionSpline

        self.X = np.asarray(X, dtype=float).ravel()
        group = np.asarray(group, dtype=str)
        self.n = len(self.X)
        self.group = group
        self.k = k
        self.levels = sorted(np.unique(group).tolist())
        n_levels = len(self.levels)

        # Fit a single global basis using all x values
        if basis == "cr":
            glob_basis = CubicRegressionSpline(self.X, k=k)
        else:
            glob_basis = PSplineBasis(self.X, k=k)

        B_global = glob_basis.B if hasattr(glob_basis, "B") else glob_basis.basis_matrix
        S_global = glob_basis.S if hasattr(glob_basis, "S") else glob_basis.penalty_matrix

        # Expand: for each level, zero out rows where group != level
        self.B = np.zeros((self.n, k * n_levels))
        for li, lv in enumerate(self.levels):
            mask = group == lv
            self.B[mask, li * k : (li + 1) * k] = B_global[mask, :]

        # Block-diagonal penalty (one block per level, same S)
        self.S = np.zeros((k * n_levels, k * n_levels))
        for li in range(n_levels):
            self.S[li * k : (li + 1) * k, li * k : (li + 1) * k] = S_global

    def basis_matrix(self) -> np.ndarray:
        return self.B

    def penalty_matrix(self) -> np.ndarray:
        return self.S


# ---------------------------------------------------------------------------
# Factor Smooth Deviations (bs='sz')
# ---------------------------------------------------------------------------


class FactorDeviation:
    """Smooth deviations per factor level (bs='sz').

    Like FactorSmooth but each level's smooth represents the *deviation*
    from the overall smooth.  Analogous to random slopes in LMMs.

    The basis includes:
      - Global smooth (k cols)
      - Per-level deviations (k * n_levels cols), penalised separately

    Attributes:
        B: Full basis matrix including global + deviations.
        S_list: List of penalty matrices; S_list[0] = global, rest = per-level.
    """

    def __init__(
        self,
        X: np.ndarray,
        group: np.ndarray,
        k: int = 10,
        basis: str = "cr",
    ) -> None:
        from pymgcv.smooth.bspline import PSplineBasis
        from pymgcv.smooth.cubic_spline import CubicRegressionSpline

        self.X = np.asarray(X, dtype=float).ravel()
        group = np.asarray(group, dtype=str)
        self.n = len(self.X)
        self.k = k
        self.levels = sorted(np.unique(group).tolist())
        n_levels = len(self.levels)

        if basis == "cr":
            glob = CubicRegressionSpline(self.X, k=k)
        else:
            glob = PSplineBasis(self.X, k=k)

        B_glob = glob.B if hasattr(glob, "B") else glob.basis_matrix
        S_glob = glob.S if hasattr(glob, "S") else glob.penalty_matrix

        # Global basis + per-level deviation blocks
        B_parts = [B_glob]
        self.S_list = [S_glob]

        total_dev_cols = k * n_levels
        B_dev = np.zeros((self.n, total_dev_cols))
        for li, lv in enumerate(self.levels):
            mask = group == lv
            B_dev[mask, li * k : (li + 1) * k] = B_glob[mask, :]
            dev_S = np.zeros((total_dev_cols, total_dev_cols))
            dev_S[li * k : (li + 1) * k, li * k : (li + 1) * k] = S_glob
            self.S_list.append(dev_S)

        self.B = np.hstack([B_glob, B_dev])
        # Combined S for single-lambda use (global only)
        p = self.B.shape[1]
        self.S = np.zeros((p, p))
        self.S[:k, :k] = S_glob

    def basis_matrix(self) -> np.ndarray:
        return self.B

    def penalty_matrices(self) -> list[np.ndarray]:
        p = self.B.shape[1]
        mats = []
        for S in self.S_list:
            full = np.zeros((p, p))
            full[: S.shape[0], : S.shape[1]] = S
            mats.append(full)
        return mats
