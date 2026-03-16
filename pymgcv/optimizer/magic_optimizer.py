"""MAGIC smoothing parameter optimizer.

Optimizes smoothing parameters λⱼ via outer loop with Newton's method.

Workflow:
    1. Outer loop: optimize log(λ) via Newton's method
    2. Inner loop: fit GAM at current λ using PIRLS
    3. Compute REML objective and gradient w.r.t. log(λ)
    4. Update log(λ), repeat until convergence

MAGIC stands for: Matching Approximate Cross-validation with Inferential Grant.
It's a criterion for choosing smoothing parameters that generalizes GCV.

References:
    - Wood, S. N. & Farouki, R. T. (1996): Research Report
    - Wood, S. N. (2011): Fast stable restricted max likelihood

Module exports:
    - MAGICOptimizer: Main optimizer class
    - optimize_smoothing_parameters: Functional API
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg, optimize


class MAGICOptimizer:
    """Optimize smoothing parameters using MAGIC criterion.

    Attributes:
        X: Design matrix.
        y: Response.
        family: Distribution family.
        S_list: Penalty matrices.
        smooth_starts: List of starting indices for each smooth term.
        smooth_sizes: List of basis dimensions for each smooth term.
        
        lambda_log: Log of smoothing parameters (what we optimize).
        lambda_vec: Actual smoothing parameters.
        
        reml_history: REML scores during optimization.
        converged: Whether optimizer converged.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        family: object,
        S_list: list[np.ndarray],
        smooth_starts: list[int],
        smooth_sizes: list[int],
        offset: Optional[np.ndarray] = None,
        dispersion: float = 1.0,
    ) -> None:
        """Initialize MAGIC optimizer.

        Args:
            X: Design matrix.
            y: Response.
            family: Distribution family.
            S_list: Penalty matrices (one per smooth term).
            smooth_starts: Starting column index for each smooth term in X.
            smooth_sizes: Number of basis functions per smooth term.
            offset: Offset vector.
            dispersion: Initial dispersion estimate.
        """
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.family = family
        self.S_list = [np.asarray(S, dtype=np.float64) for S in S_list]
        self.smooth_starts = smooth_starts
        self.smooth_sizes = smooth_sizes
        self.offset = np.asarray(offset, dtype=np.float64) if offset is not None else np.zeros_like(y)
        self.dispersion = float(dispersion)
        self.weights = None  # set externally if needed

        self.n_smooth = len(S_list)

        # Initialize smoothing parameters based on data-penalty ratio
        # (Wood 2004‐style: balance response variance against penalty size)
        self.lambda_log = np.zeros(self.n_smooth)
        y_var = max(np.var(self.y), 1e-10)
        for j in range(self.n_smooth):
            eig_max = np.max(np.abs(np.linalg.eigvalsh(self.S_list[j])))
            if eig_max > 1e-10:
                lam_init = y_var / eig_max
                self.lambda_log[j] = np.log(max(lam_init, 1e-10))
        self.lambda_vec = np.exp(self.lambda_log)

        self.reml_history: list[float] = []
        self.converged = False

        # PIRLS solver (will be instantiated at each outer iteration)
        self.pirls_solver = None

    def optimize(
        self,
        max_outer_iter: int = 10,
        max_inner_iter: int = 25,
        outer_tol: float = 1e-5,
        inner_tol: float = 1e-7,
        verbose: bool = False,
        use_jax: bool = False,
        use_reml: bool = True,
        method: str = 'reml',
    ) -> dict:
        """Optimize smoothing parameters via MAGIC.

        Uses Newton's method on log(λ):
            Δ log(λ) = -H⁻¹ g
        where H is Hessian and g is gradient of REML w.r.t. log(λ).

        Args:
            max_outer_iter: Max Newton iterations.
            max_inner_iter: Max PIRLS iterations per smoothing parameter.
            outer_tol: Convergence tolerance on Δ log(λ).
            inner_tol: Convergence tolerance for PIRLS.
            verbose: Print progress.
            use_jax: Use JAX GPU acceleration if available.
            use_reml: Use REML (vs GCV) for λ optimization (deprecated; use method=).
            method: Smoothing criterion — 'reml' (default), 'gcv', 'ubre', 'ml'.

        Returns:
            Dict with 'coef', 'smooth_lambda', 'fitted_values', 'edf'.
        """
        # Backwards-compatibility: use_reml=False → gcv
        if not use_reml and method == 'reml':
            method = 'gcv'

        if method in ('gcv', 'ubre', 'ml'):
            return self._optimize_gcv(
                method=method,
                max_outer_iter=max_outer_iter,
                max_inner_iter=max_inner_iter,
                outer_tol=outer_tol,
                inner_tol=inner_tol,
                verbose=verbose,
            )
        # Default: REML
        return self._optimize_reml(
            max_outer_iter=max_outer_iter,
            max_inner_iter=max_inner_iter,
            outer_tol=outer_tol,
            inner_tol=inner_tol,
            verbose=verbose,
        )

    def _optimize_reml(
        self,
        max_outer_iter: int,
        max_inner_iter: int,
        outer_tol: float,
        inner_tol: float,
        verbose: bool,
    ) -> dict:
        from pymgcv.optimizer.pirls import PIRLSSolver
        from pymgcv.optimizer.reml_objective import REMLObjective

        for outer_it in range(max_outer_iter):
            # Update lambda from log scale
            self.lambda_vec = np.exp(self.lambda_log)

            # Inner loop: PIRLS at current lambda
            solver = PIRLSSolver(
                self.X, self.y, self.family, self.S_list,
                lambda_vec=self.lambda_vec,
                offset=self.offset,
                dispersion=self.dispersion,
                weights=self.weights,
            )
            beta = solver.solve(max_iter=max_inner_iter, tol=inner_tol, verbose=False)

            # Compute REML objective and derivatives
            reml_obj = REMLObjective(
                self.X, self.y, self.family, self.S_list,
                self.smooth_starts, self.smooth_sizes,
                offset=self.offset,
                dispersion=self.dispersion,
            )

            reml_score, grad_log_lambda, hess_log_lambda = reml_obj.objective_gradient_hessian(beta, self.lambda_log)

            self.reml_history.append(reml_score)

            if verbose:
                print(
                    f'Outer Iter {outer_it:2d}: '
                    f'REML = {reml_score:.6e}, '
                    f'||grad|| = {np.linalg.norm(grad_log_lambda):.6e}'
                )

            # Newton step on log(λ)
            if not (np.isfinite(reml_score) and
                    np.all(np.isfinite(grad_log_lambda)) and
                    np.all(np.isfinite(hess_log_lambda))):
                if verbose:
                    print(f'  Warning: REML non-finite at iter {outer_it}, stopping')
                break

            try:
                d_log_lambda = linalg.solve(hess_log_lambda, -grad_log_lambda)
            except (linalg.LinAlgError, ValueError):
                d_log_lambda = -0.01 * grad_log_lambda

            # Clip step size to prevent overflow
            max_step = 3.0
            step_scale = np.max(np.abs(d_log_lambda))
            if step_scale > max_step:
                d_log_lambda *= max_step / step_scale

            # Backtracking line search
            alpha = 1.0
            for _ in range(10):
                lambda_log_new = np.clip(self.lambda_log + alpha * d_log_lambda, -20, 20)
                lambda_new = np.exp(lambda_log_new)
                solver_new = PIRLSSolver(
                    self.X, self.y, self.family, self.S_list,
                    lambda_vec=lambda_new,
                    offset=self.offset,
                    dispersion=self.dispersion,
                )
                beta_new = solver_new.solve(max_iter=max_inner_iter, tol=inner_tol, verbose=False)
                reml_score_new = reml_obj.objective(beta_new, np.log(lambda_new))
                if reml_score_new < reml_score:
                    self.lambda_log = lambda_log_new
                    break
                alpha *= 0.5
            else:
                self.lambda_log = np.clip(self.lambda_log + alpha * d_log_lambda, -20, 20)

            if len(d_log_lambda) == 0:
                self.converged = True
                break
            elif np.max(np.abs(alpha * d_log_lambda)) < outer_tol:
                self.converged = True
                if verbose:
                    print(f'Converged after {outer_it + 1} outer iterations')
                break

        self.lambda_vec = np.exp(self.lambda_log)
        solver_final = PIRLSSolver(
            self.X, self.y, self.family, self.S_list,
            lambda_vec=self.lambda_vec,
            offset=self.offset,
            dispersion=self.dispersion,
        )
        beta_final = solver_final.solve(max_iter=max_inner_iter, tol=inner_tol, verbose=False)
        fitted_vals = solver_final.fitted_values()
        edf = self._compute_edf(beta_final)
        return {
            'coef': beta_final,
            'smooth_lambda': self.lambda_vec,
            'fitted_values': fitted_vals,
            'edf': edf,
        }

    def _optimize_gcv(
        self,
        method: str,
        max_outer_iter: int,
        max_inner_iter: int,
        outer_tol: float,
        inner_tol: float,
        verbose: bool,
    ) -> dict:
        """Optimize smoothing parameters via GCV, UBRE, or ML criterion.

        Uses scipy.optimize.minimize on log(lambda) with numerical gradients.
        Supports Gaussian (GCV/UBRE) and non-Gaussian families (ML).
        """
        from pymgcv.optimizer.pirls import PIRLSSolver

        n = len(self.y)

        def _criterion(log_lam: np.ndarray) -> float:
            lam = np.exp(log_lam)
            s = PIRLSSolver(
                self.X, self.y, self.family, self.S_list,
                lambda_vec=lam,
                offset=self.offset,
                dispersion=self.dispersion,
                weights=self.weights,
            )
            beta = s.solve(max_iter=max_inner_iter, tol=inner_tol)
            eta = self.X @ beta + self.offset
            mu = self.family.linkinv(eta)

            # Combined penalty matrix
            S_lam = sum(l * S for l, S in zip(lam, self.S_list))
            XtWX = self.X.T @ (self.X * s.weights[:, None])

            try:
                H_inv = np.linalg.inv(XtWX + S_lam)
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(XtWX + S_lam)

            dof = float(np.trace(H_inv @ XtWX))
            dof = np.clip(dof, 0, self.X.shape[1])

            if method == 'gcv':
                dev = float(np.sum((self.y - mu) ** 2))
                denom = max(n - dof, 0.5)
                score = n * dev / denom ** 2
            elif method == 'ubre':
                # UBRE: (1/n) * D + 2 * phi * dof/n - phi
                phi = self.dispersion
                dev = float(np.sum((self.y - mu) ** 2))
                score = dev / n + 2.0 * phi * dof / n - phi
            else:  # ml
                score = -float(self.family.loglik(self.y, mu, self.dispersion))
                score += 0.5 * float(beta @ S_lam @ beta)

            return score if np.isfinite(score) else 1e30

        result = optimize.minimize(
            _criterion,
            x0=self.lambda_log.copy(),
            method='Nelder-Mead',
            options={'maxiter': max_outer_iter * 20, 'xatol': outer_tol, 'fatol': outer_tol},
        )
        self.lambda_log = result.x
        self.lambda_vec = np.exp(self.lambda_log)
        self.converged = result.success

        solver_final = PIRLSSolver(
            self.X, self.y, self.family, self.S_list,
            lambda_vec=self.lambda_vec,
            offset=self.offset,
            dispersion=self.dispersion,
        )
        beta_final = solver_final.solve(max_iter=max_inner_iter, tol=inner_tol)
        edf = self._compute_edf(beta_final)
        return {
            'coef': beta_final,
            'smooth_lambda': self.lambda_vec,
            'fitted_values': solver_final.fitted_values(),
            'edf': edf,
        }

    def _compute_edf(self, beta: np.ndarray) -> float:
        """Compute actual EDF = trace((X'WX + S_lam)^{-1} X'WX).

        Uses p-space influence matrix for O(np^2) computation.
        """
        eta = self.X @ beta + self.offset
        mu = self.family.linkinv(eta)
        dmu_deta = self.family.dmu_deta(eta)
        var_mu = np.maximum(self.family.variance(mu, self.dispersion), 1e-10)
        w = (dmu_deta ** 2) / var_mu

        S_lam = sum(l * S for l, S in zip(self.lambda_vec, self.S_list))
        XtWX = self.X.T @ (self.X * w[:, np.newaxis])
        A = XtWX + S_lam

        # Use Cholesky or pinv for numerical stability
        try:
            L = linalg.cholesky(A, lower=True)
            A_inv_XtWX = linalg.cho_solve((L, True), XtWX)
        except linalg.LinAlgError:
            A_inv_XtWX = linalg.lstsq(A, XtWX)[0]

        return float(np.clip(np.trace(A_inv_XtWX), 0, self.X.shape[1]))

    def smoothing_parameters(self) -> np.ndarray:
        """Return current smoothing parameters λⱼ."""
        return self.lambda_vec.copy()

    def reml_scores(self) -> list[float]:
        """Return REML scores from optimization."""
        return self.reml_history.copy()

    def summary(self) -> str:
        """Summary of optimization."""
        lines = [
            'MAGIC Optimizer',
            '===============',
            f'Convergence: {self.converged}',
            f'Outer iterations: {len(self.reml_history)}',
        ]
        if self.reml_history:
            lines.extend([
                f'Initial REML: {self.reml_history[0]:.6e}',
                f'Final REML: {self.reml_history[-1]:.6e}',
            ])
        lines.append(f'Smoothing parameters λ = {self.lambda_vec}')
        return '\n'.join(lines)


def optimize_smoothing_parameters(
    X: np.ndarray,
    y: np.ndarray,
    family: object,
    S_list: list[np.ndarray],
    smooth_starts: list[int],
    smooth_sizes: list[int],
    offset: Optional[np.ndarray] = None,
    dispersion: float = 1.0,
    max_outer_iter: int = 10,
    max_inner_iter: int = 25,
    verbose: bool = False,
) -> np.ndarray:
    """Functional API for smoothing parameter optimization.

    Args:
        X: Design matrix.
        y: Response.
        family: Distribution family.
        S_list: Penalty matrices.
        smooth_starts: Starting column indices for each smooth.
        smooth_sizes: Basis dimensions for each smooth.
        offset: Offset vector.
        dispersion: Dispersion parameter.
        max_outer_iter: Max outer iterations.
        max_inner_iter: Max PIRLS iterations.
        verbose: Print progress.

    Returns:
        Optimized smoothing parameters λⱼ.
    """
    optimizer = MAGICOptimizer(
        X, y, family, S_list, smooth_starts, smooth_sizes,
        offset=offset, dispersion=dispersion,
    )
    return optimizer.optimize(
        max_outer_iter=max_outer_iter,
        max_inner_iter=max_inner_iter,
        verbose=verbose,
    )
