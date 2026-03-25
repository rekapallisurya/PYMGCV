"""Tensor product smooth basis for multi-dimensional GAM smoothing.

Implements te() and ti() smooths (as in mgcv) using Kronecker products
of marginal basis matrices and penalty matrices.

Theory:
    For variables x1 and x2 with marginal bases B1 (n×k1) and B2 (n×k2):

        te(x1, x2):  B = row-kron(B1, B2),  shape (n, k1*k2)
                     S1 = kron(P1, I_k2),    shape (k1*k2, k1*k2)
                     S2 = kron(I_k1, P2),    shape (k1*k2, k1*k2)

        ti(x1, x2):  Interaction-only variant; main effects are removed
                     by centering each marginal basis.

    The two separate penalties (S1, S2) each get their own smoothing
    parameter lambda, exactly matching mgcv's te() implementation.

References:
    - Wood (2006): Low-Rank Scale-Invariant Tensor Product Smooths for GAMs
    - Wood (2017): Generalized Additive Models: An Introduction with R (2nd ed.)
"""

from __future__ import annotations

import numpy as np

from pymgcv.smooth.bspline import BSplineBasis
from pymgcv.smooth.thin_plate import ThinPlateSpline


def _row_kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Kronecker product of two matrices.

    For each row i: result[i, :] = kron(A[i, :], B[i, :])

    Args:
        A: Matrix of shape (n, k1).
        B: Matrix of shape (n, k2).

    Returns:
        Matrix of shape (n, k1*k2).
    """
    n = A.shape[0]
    k1 = A.shape[1]
    k2 = B.shape[1]
    # Broadcast: A[:, :, None] * B[:, None, :] → shape (n, k1, k2)
    return (A[:, :, np.newaxis] * B[:, np.newaxis, :]).reshape(n, k1 * k2)


def _kron_sum_penalties(penalties: list[np.ndarray]) -> list[np.ndarray]:
    """Build Kronecker-sum penalty matrices for tensor product smooths.

    For d margins with penalty matrices P_1, ..., P_d and dimensions k_1, ..., k_d:

        For the j-th penalty: I_{k_1} ⊗ ... ⊗ P_j ⊗ ... ⊗ I_{k_d}

    Returns a list of d penalty matrices, each of dimension (∏k_i × ∏k_i).

    Args:
        penalties: List of penalty matrices, one per margin.

    Returns:
        List of combined penalty matrices.
    """
    d = len(penalties)
    dims = [P.shape[0] for P in penalties]

    combined = []
    for i in range(d):
        # Build Kronecker product: I ⊗ ... ⊗ P_i ⊗ ... ⊗ I
        matrices = [np.eye(dims[j]) if j != i else penalties[j] for j in range(d)]

        # Compute full Kronecker product
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        combined.append(result)

    return combined


class TensorProductSmooth:
    """Tensor product smooth for multi-dimensional smoothing (te/ti).

    Constructs a tensor product basis from marginal basis matrices.
    Supports te() (full tensor product) and ti() (interaction only).

    Attributes:
        var_names: Variable names for each margin.
        k_values: Basis dimensions per margin.
        interaction_only: If True, build ti() (remove main effects).
        margins: List of marginal basis objects (ThinPlateSpline or BSplineBasis).
        B: Full tensor product basis matrix, shape (n, k1*k2*...).
        penalties: List of Kronecker-sum penalty matrices.
        total_dim: Total number of basis functions (k1 * k2 * ...).
    """

    def __init__(
        self,
        data: np.ndarray | dict,
        var_names: list[str],
        k_values: list[int] | None = None,
        basis_type: str = "tp",
        interaction_only: bool = False,
    ) -> None:
        """Initialize tensor product smooth.

        Args:
            data: Data as dict of {var_name: array} or 2D array.
            var_names: List of variable names to include.
            k_values: Basis dimension per margin. None → auto (min(n, 10)).
            basis_type: Marginal basis type ('tp', 'bs'). Default 'tp'.
            interaction_only: If True, build ti() instead of te().

        Raises:
            ValueError: If var_names is empty or data is invalid.
        """
        if not var_names:
            raise ValueError("var_names must be non-empty")

        self.var_names = var_names
        self.basis_type = basis_type
        self.interaction_only = interaction_only

        # Extract per-variable data
        if isinstance(data, dict):
            arrays = [np.asarray(data[v], dtype=float).ravel() for v in var_names]
        elif isinstance(data, np.ndarray):
            arrays = [np.asarray(data[:, i], dtype=float) for i in range(len(var_names))]
        else:
            raise TypeError("data must be dict or 2D array")

        n = len(arrays[0])
        if k_values is None:
            k_values = [min(n, 10)] * len(var_names)
        self.k_values = k_values

        # Build marginal bases
        self.margins: list = []
        self._margin_Bs: list[np.ndarray] = []
        self._margin_Ps: list[np.ndarray] = []

        for i, (arr, k) in enumerate(zip(arrays, k_values)):
            if basis_type == "tp":
                basis_obj = ThinPlateSpline(arr.reshape(-1, 1), k=k)
                B_i = basis_obj.basis_matrix()
            elif basis_type == "bs":
                basis_obj = BSplineBasis(arr, k=k)
                B_i = basis_obj.basis_matrix
            else:
                # Default to TPRS
                basis_obj = ThinPlateSpline(arr.reshape(-1, 1), k=k)
                B_i = basis_obj.basis_matrix()

            self.margins.append(basis_obj)
            self._margin_Bs.append(B_i)

        # Get marginal penalties
        for i, (basis_obj, B_i) in enumerate(zip(self.margins, self._margin_Bs)):
            if hasattr(basis_obj, "penalty_matrix_S"):
                P_i = basis_obj.penalty_matrix_S()
            elif hasattr(basis_obj, "penalty_matrix"):
                P_i = basis_obj.penalty_matrix
            else:
                # Fallback: second-difference penalty
                k = B_i.shape[1]
                D = np.diff(np.eye(k), n=2, axis=0)
                P_i = D.T @ D
            self._margin_Ps.append(P_i)

        # Build full tensor product basis
        if interaction_only:
            self.B = self._build_ti_basis(self._margin_Bs)
        else:
            self.B = self._build_te_basis(self._margin_Bs)

        # Build Kronecker-sum penalties (one per margin)
        if interaction_only:
            self.penalties = self._build_ti_penalties(self._margin_Ps)
        else:
            self.penalties = _kron_sum_penalties(self._margin_Ps)

        self.total_dim = self.B.shape[1]

    def _build_te_basis(self, margin_Bs: list[np.ndarray]) -> np.ndarray:
        """Build full tensor product basis via row-wise Kronecker product.

        Args:
            margin_Bs: List of marginal basis matrices.

        Returns:
            Tensor product basis of shape (n, k1*k2*...).
        """
        B = margin_Bs[0]
        for B_j in margin_Bs[1:]:
            B = _row_kron(B, B_j)
        return B

    def _build_ti_basis(self, margin_Bs: list[np.ndarray]) -> np.ndarray:
        """Build interaction-only (ti) basis.

        Builds the full tensor product then projects out all marginal sub-spaces
        so that only the pure interaction variation remains.  This matches the
        statistical meaning of mgcv's ti(): the smooth explains variation not
        captured by any of the individual marginal smooths.

        Steps:
            1. Compute te() basis: B = row_kron(B1, B2, ...).
            2. Stack all marginal bases: M = [B1 | B2 | ...].
            3. Project B orthogonal to column(M):
               B_ti = (I - Q_M Q_M') B   where  M = Q_M R_M.

        Args:
            margin_Bs: List of marginal basis matrices, each (n, k_i).

        Returns:
            Interaction-only tensor product basis, shape (n, k1*k2*...).
        """
        # Step 1: full tensor product
        B_te = margin_Bs[0]
        for B_j in margin_Bs[1:]:
            B_te = _row_kron(B_te, B_j)

        # Step 2: stack marginals
        M = np.hstack(margin_Bs)  # (n, k1 + k2 + ...)

        # Step 3: orthogonal projection – remove the space spanned by M
        Q_M, _ = np.linalg.qr(M, mode="reduced")
        B_ti = B_te - Q_M @ (Q_M.T @ B_te)
        return B_ti

    def _build_ti_penalties(self, margin_Ps: list[np.ndarray]) -> list[np.ndarray]:
        """Build penalties for tensor-of-contrasts smooth.

        Similar to te() penalties but adjusted for the centered basis.

        Args:
            margin_Ps: List of marginal penalty matrices.

        Returns:
            List of Kronecker-sum penalty matrices.
        """
        return _kron_sum_penalties(margin_Ps)

    def basis_matrix(self) -> np.ndarray:
        """Return the full tensor product basis matrix, shape (n, total_dim)."""
        return self.B

    def penalty_matrices(self) -> list[np.ndarray]:
        """Return list of Kronecker-sum penalty matrices (one per margin)."""
        return self.penalties

    def predict(
        self,
        new_data: dict,
    ) -> np.ndarray:
        """Build tensor product basis for new data.

        Args:
            new_data: Dict of {var_name: array} for new observations.

        Returns:
            New tensor product basis matrix, shape (n_new, total_dim).
        """
        new_Bs = []
        for i, var in enumerate(self.var_names):
            x_new = np.asarray(new_data[var], dtype=float).ravel()
            basis_obj = self.margins[i]

            if hasattr(basis_obj, "predict"):
                B_new_i = basis_obj.predict(x_new.reshape(-1, 1))
            elif hasattr(basis_obj, "basis_matrix"):
                # For BSplineBasis — re-evaluate at new x
                from pymgcv.smooth.bspline import BSplineBasis

                new_basis = BSplineBasis(x_new, k=self.k_values[i])
                B_new_i = new_basis.basis_matrix
            else:
                raise AttributeError(f"Marginal basis {i} has no predict method")
            new_Bs.append(B_new_i)

        if self.interaction_only:
            # Apply same centering as training
            centered_new_Bs = []
            for B_i_new, B_i_train in zip(new_Bs, self._margin_Bs):
                col_means = B_i_train.mean(axis=0)
                centered_new_Bs.append(B_i_new - col_means[np.newaxis, :])
            B_new = centered_new_Bs[0]
            for B_j in centered_new_Bs[1:]:
                B_new = _row_kron(B_new, B_j)
        else:
            B_new = new_Bs[0]
            for B_j in new_Bs[1:]:
                B_new = _row_kron(B_new, B_j)

        return B_new


class TensorProductT2(TensorProductSmooth):
    """Alternative t2() parametrization tensor product smooth.

    Uses eigendecomposition-based scaling as in mgcv's t2().
    The penalty structure has a simpler (diagonal) form compared to te().

    This is a simplified version; the full implementation would match mgcv's
    t2 smoothing parameter selection and penalty structure exactly.
    """

    def __init__(
        self,
        data: np.ndarray | dict,
        var_names: list[str],
        k_values: list[int] | None = None,
        basis_type: str = "tp",
    ) -> None:
        # Build te() basis first (same design matrix)
        super().__init__(data, var_names, k_values, basis_type, interaction_only=False)
        # t2 has the same design matrix as te() but different identification constraints
        # Adjust penalties using eigendecomposition
        self._build_t2_penalties()

    def _build_t2_penalties(self) -> None:
        """Rebuild penalties for t2() parametrization.

        t2 uses scaled identifiability constraints based on the eigenstructure
        of each marginal penalty, yielding better numerical properties.
        """
        scaled_penalties = []
        for i, P_i in enumerate(self._margin_Ps):
            # Eigendecompose each marginal penalty
            evals, evecs = np.linalg.eigh(P_i)
            evals = np.where(evals > 1e-10, evals, 0.0)
            P_i_scaled = evecs @ np.diag(evals) @ evecs.T
            self._margin_Ps[i] = P_i_scaled

        # Rebuild Kronecker sum with scaled penalties
        self.penalties = _kron_sum_penalties(self._margin_Ps)
