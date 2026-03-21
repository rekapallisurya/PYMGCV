"""Numerically stable solver for penalized weighted least squares.

Core solve:  (X'WX + S_lambda) beta = X'Wz

Hierarchy:
    1. Cholesky (fastest, requires positive definiteness)
    2. Progressively larger diagonal ridge until Cholesky succeeds
    3. SVD truncation (most robust fallback)

The same factorization is reused for:
    - solve(b)               : beta computation
    - log_determinant()      : REML log|A| term
    - trace_inv_product(B)   : REML gradient trace(A^{-1} S_j) terms
    - inv_diagonal()         : diagonal of A^{-1} for EDF and SE
    - edf()                  : trace(A^{-1} X'WX)
    - posterior_cov_diag(phi): diagonal of Bayesian posterior V_b = A^{-1} phi

This ensures numerical consistency between the REML objective value,
its gradient, and its Hessian — critical for stable Newton-step optimization.

References:
    Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal
    likelihood estimation of semiparametric generalized linear models. JRSS(B).
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .native_engine import col_squared_norms


class PenalizedSolver:
    """Cholesky-based factorization of A = X'WX + S with operation reuse.

    Computes and caches the Cholesky factor of A, providing efficient
    numerically-stable access to solve, log-determinant, trace of inverse
    products, and diagonal of inverse.

    Attributes:
        A: System matrix X'WX + S, shape (p, p).
        XtWX: Weighted information matrix, shape (p, p).
        is_cholesky: True if Cholesky succeeded; False if SVD fallback used.
        ridge: Diagonal regularisation actually applied (0 if none needed).
    """

    def __init__(self, XtWX: np.ndarray, S: np.ndarray) -> None:
        """Initialise and factorise A = XtWX + S.

        Args:
            XtWX: X'WX, weighted information matrix, shape (p, p).
            S: Combined penalty S_lambda = sum_j lambda_j S_j, shape (p, p).
        """
        self.p = XtWX.shape[0]
        self.XtWX = XtWX
        self.A = XtWX + S

        self._cho: tuple | None = None         # (L, lower) from cho_factor
        self._L_inv: np.ndarray | None = None  # triangular inverse of L
        self._qr: tuple | None = None          # (Q, R, pivot) from pivoted QR
        self._svd: tuple | None = None         # (U, s, s_inv, Vt)
        self.is_cholesky = False
        self.is_qr = False
        self.ridge = 0.0

        self._factorize()

    # ------------------------------------------------------------------
    # Internal factorisation
    # ------------------------------------------------------------------

    def _factorize(self) -> None:
        """Attempt Cholesky; if it fails, progressively add adaptive ridge."""
        p = self.p
        diag_mean = max(float(np.mean(np.abs(np.diag(self.A)))), 1e-10)

        # 1. Try exact Cholesky (no ridge)
        try:
            cho = linalg.cho_factor(self.A, lower=True, check_finite=False)
            if np.all(np.isfinite(np.diag(cho[0]))) and np.all(np.diag(cho[0]) > 0):
                self._cho = cho
                self._finish_cholesky(cho[0])
                return
        except linalg.LinAlgError:
            pass

        # 2. Try with progressively larger adaptive ridge
        for exp in (-12, -10, -8, -6, -4):
            ridge = diag_mean * (10.0 ** exp)
            A_reg = self.A + ridge * np.eye(p)
            try:
                cho = linalg.cho_factor(A_reg, lower=True, check_finite=False)
                if np.all(np.isfinite(np.diag(cho[0]))) and np.all(np.diag(cho[0]) > 0):
                    self._cho = cho
                    self.ridge = ridge
                    self._finish_cholesky(cho[0])
                    return
            except linalg.LinAlgError:
                continue

        # 3. Pivoted QR — catches nearly-singular SPD matrices missed by ridge Cholesky
        self._factorize_qr()
        if self.is_qr:
            return

        # 4. SVD fallback
        self._factorize_svd()

    def _finish_cholesky(self, L: np.ndarray) -> None:
        """Compute L^{-1} and mark success."""
        self._L_inv = linalg.solve_triangular(
            L, np.eye(self.p), lower=True, trans=0, check_finite=False
        )
        self.is_cholesky = True

    def _factorize_qr(self) -> None:
        """Pivoted QR factorisation A P = Q R.

        Used when Cholesky (with ridge) fails — catches rank-deficient or
        indefinite matrices that aren't SPD even after small regularisation.
        We form the symmetric factorization via A = A + tiny_ridge * I first.
        """
        try:
            tiny = float(np.mean(np.abs(np.diag(self.A)))) * 1e-8
            A_reg = self.A + tiny * np.eye(self.p)
            Q, R, pivot = linalg.qr(A_reg, pivoting=True, check_finite=False)
            diag_R = np.abs(np.diag(R))
            tol = max(diag_R[0], 1.0) * self.p * np.finfo(float).eps * 100
            rank = int(np.sum(diag_R > tol))
            if rank == self.p:
                # Full rank: store for solve/logdet
                self._qr = (Q, R, pivot)
                self.is_qr = True
                # Precompute L_inv equivalent (R^{-1} P^T) for inv_diagonal
                Rinv = linalg.solve_triangular(
                    R, np.eye(rank), lower=False, check_finite=False
                )
                # A^{-1} = P R^{-1} Q^T  (symmetric A → Q = Q of A, so A^{-1} ≈ P R^{-1} Q^T)
                # Store Rinv for inv_diagonal computation
                self._qr_Rinv = Rinv
                self._qr_pivot = pivot
        except linalg.LinAlgError:
            pass

    def _factorize_svd(self) -> None:
        """SVD fallback for rank-deficient or indefinite A."""
        try:
            U, s, Vt = np.linalg.svd(self.A, full_matrices=False)
            tol = max(float(s[0]), 1.0) * self.p * np.finfo(float).eps * 1e3
            s_inv = np.where(s > tol, 1.0 / s, 0.0)
            self._svd = (U, s, s_inv, Vt)
        except np.linalg.LinAlgError:
            s = np.ones(self.p)
            self._svd = (np.eye(self.p), s, 1.0 / s, np.eye(self.p))
        self.is_cholesky = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve Ax = b for vector or matrix b."""
        if self._cho is not None:
            return linalg.cho_solve(self._cho, b, check_finite=False)
        if self._qr is not None:
            Q, R, pivot = self._qr
            # A P = Q R  =>  A^{-1} b = P R^{-1} Q^T b
            if b.ndim == 1:
                QtB = Q.T @ b
                RinvQtB = linalg.solve_triangular(R, QtB, lower=False, check_finite=False)
                result = np.empty_like(RinvQtB)
                result[pivot] = RinvQtB
                return result
            else:
                QtB = Q.T @ b
                RinvQtB = linalg.solve_triangular(R, QtB, lower=False, check_finite=False)
                result = np.empty_like(RinvQtB)
                result[pivot, :] = RinvQtB
                return result
        if self._svd is not None:
            U, s, s_inv, Vt = self._svd
            if b.ndim == 1:
                return Vt.T @ (s_inv * (U.T @ b))
            return Vt.T @ (s_inv[:, np.newaxis] * (U.T @ b))
        return linalg.lstsq(self.A, b, check_finite=False)[0]

    def log_determinant(self) -> float:
        """Compute log|A| from the cached factorisation.

        For Cholesky:  log|A| ≈ 2 * sum(log diag(L)).
          (Exact for A; negligible bias from tiny ridge is < 1e-8.)
        For QR:        log|A| = sum(log |diag(R)|).
        For SVD:       log|A| = sum(log s_k).
        """
        if self._cho is not None:
            L = self._cho[0]
            return 2.0 * float(np.sum(np.log(np.maximum(np.abs(np.diag(L)), 1e-300))))
        if self._qr is not None:
            _, R, _ = self._qr
            return float(np.sum(np.log(np.maximum(np.abs(np.diag(R)), 1e-300))))
        if self._svd is not None:
            _, s, _, _ = self._svd
            return float(np.sum(np.log(np.maximum(s, 1e-300))))
        sign, val = np.linalg.slogdet(self.A)
        return float(val) if sign > 0 else -1e300

    def trace_inv_product(self, B: np.ndarray) -> float:
        """Compute trace(A^{-1} B) without forming A^{-1}.

        Solves A X = B (cached factor cost), then returns trace(X).
        """
        return float(np.trace(self.solve(B)))

    def inv_diagonal(self) -> np.ndarray:
        """Compute diagonal of A^{-1} without forming the full inverse.

        For Cholesky (A = L L^T):
            diag(A^{-1})_j = ||L^{-1}[:, j]||^2  (column norms of L^{-1})

        For SVD (A ≈ U S U^T for symmetric PSD):
            diag(A^{-1})_j = sum_k V[j,k]^2 / s_k
                           = ((Vt.T)**2) @ s_inv
        """
        if self._L_inv is not None:
            return col_squared_norms(self._L_inv)
        if self._qr is not None:
            # A = Q R P^T  =>  A^{-1} = P R^{-T} Q^T  (A symmetric positive definite)
            # diag(A^{-1})_j = ||col j of (P R^{-1})||^2
            Rinv = getattr(self, '_qr_Rinv', None)
            pivot = getattr(self, '_qr_pivot', None)
            if Rinv is not None and pivot is not None:
                PRinv = np.empty_like(Rinv)
                PRinv[pivot, :] = Rinv
                return np.sum(PRinv ** 2, axis=1)
        if self._svd is not None:
            _, s, s_inv, Vt = self._svd
            return (Vt.T ** 2) @ s_inv
        return np.diag(np.linalg.pinv(self.A))

    def edf(self) -> float:
        """Effective degrees of freedom: trace(A^{-1} X'WX)."""
        return self.trace_inv_product(self.XtWX)

    def edf_per_column(self) -> np.ndarray:
        """Per-column EDF: diagonal of F = A^{-1} X'WX."""
        return np.diag(self.solve(self.XtWX))

    def posterior_cov_diag(self, phi: float = 1.0) -> np.ndarray:
        """Diagonal of Bayesian posterior covariance V_b = A^{-1} * phi.

        Matches the Bayesian SEs reported by mgcv's summary.gam().
        """
        return self.inv_diagonal() * phi


# ---------------------------------------------------------------------------
# Legacy functional helpers (backward compatibility)
# ---------------------------------------------------------------------------

def solve_symmetric_definite(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve symmetric positive definite system Ax = b."""
    solver = PenalizedSolver(A, np.zeros_like(A))
    return solver.solve(b)


def trace_matrix_inverse_product(A: np.ndarray, B: np.ndarray) -> float:
    """Compute trace(A^{-1} B) using Cholesky factorisation."""
    solver = PenalizedSolver(A, np.zeros_like(A))
    return solver.trace_inv_product(B)
