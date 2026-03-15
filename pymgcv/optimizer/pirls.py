"""Penalized Iteratively Reweighted Least Squares (PIRLS) solver.

Solves the GAM fitting problem iteratively via PIRLS:

    At each iteration t:
        1. Compute predicted mean: μ = g⁻¹(X β)
        2. Compute weights: w = (dμ/dη)² / Var(Y)
        3. Compute working vector: z = X β + (y - μ) / dμ/dη
        4. Solve: (X^T W X + Sλ) β = X^T W z
        5. Repeat until convergence

References:
    - Wood, S. N. (2017): Generalized Additive Models, Ch. 3-4
    - McCullagh, P. & Nelder, J. (1989): Generalized Linear Models

Module exports:
    - PIRLSSolver: Main PIRLS solver class
    - solve_pirls: Functional API
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg


class PIRLSSolver:
    """Solve GAM via Penalized Iteratively Reweighted Least Squares.

    Attributes:
        X: Design matrix, shape (n, p).
        y: Response vector, shape (n,).
        family: Distribution family (Gaussian, Poisson, etc.).
        S_list: Penalty matrices.
        lambda_vec: Smoothing parameters.
        offset: Offset vector.
        dispersion: Dispersion parameter φ.
        
        beta: Current/final coefficient estimate.
        mu: Current/final predicted mean.
        eta: Current/final linear predictor.
        converged: Whether optimization converged.
        iterations: Number of iterations until convergence.
        history: Convergence history.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        family: object,
        S_list: list[np.ndarray],
        lambda_vec: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        dispersion: float = 1.0,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize PIRLS solver.

        Args:
            X: Design matrix, shape (n, p).
            y: Response vector, shape (n,).
            family: Distribution family object.
            S_list: Penalty matrices.
            lambda_vec: Smoothing parameters.
            offset: Offset vector.
            dispersion: Dispersion parameter.
            weights: Observation weights (optional).
        """
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.family = family
        self.S_list = [np.asarray(S, dtype=np.float64) for S in S_list]
        self.offset = self._validate_offset(offset, len(y))
        self.weights = self._validate_weights(weights, len(y))
        self.dispersion = float(dispersion)

        if lambda_vec is None:
            lambda_vec = np.ones(len(S_list))
        self.lambda_vec = np.asarray(lambda_vec, dtype=np.float64)

        self.n, self.p = self.X.shape

        # Initialize coefficients
        self.beta = np.zeros(self.p)
        self.mu: np.ndarray = np.ones_like(self.y)  # Placeholder
        self.eta: np.ndarray = np.zeros_like(self.y)

        # Convergence tracking
        self.converged = False
        self.iterations = 0
        self.history: list[dict] = []
        self.dev_history: list[float] = []  # Deviance history for monitoring

        # Construct combined penalty
        self.S = self._construct_combined_penalty()

    def _validate_offset(self, offset: Optional[np.ndarray], n: int) -> np.ndarray:
        """Validate and handle offset parameter."""
        if offset is None:
            return np.zeros(n)
        
        offset = np.asarray(offset, dtype=np.float64)
        
        if len(offset) != n:
            raise ValueError(f"Offset length {len(offset)} != n={n}")
        
        # Handle infinite offsets
        if not np.all(np.isfinite(offset)):
            offset = np.where(np.isfinite(offset), offset, 0.0)
        
        return offset

    def _validate_weights(self, weights: Optional[np.ndarray], n: int) -> np.ndarray:
        """Validate and handle weights parameter."""
        if weights is None:
            return np.ones(n)
        
        weights = np.asarray(weights, dtype=np.float64)
        
        if len(weights) != n:
            raise ValueError(f"Weights length {len(weights)} != n={n}")
        
        # Ensure positive weights
        if np.any(weights <= 0):
            raise ValueError("Weights must be positive")
        
        # Handle infinite weights
        if not np.all(np.isfinite(weights)):
            weights = np.where(np.isfinite(weights), weights, 1.0)
        
        return weights

    def _construct_combined_penalty(self) -> np.ndarray:
        """Construct combined penalty Sλ = Σⱼ λⱼ Sⱼ."""
        S = np.zeros((self.p, self.p))
        for S_j, lambda_j in zip(self.S_list, self.lambda_vec):
            S += lambda_j * S_j
        return S

    def solve(
        self,
        max_iter: int = 25,
        tol: float = 1e-7,
        verbose: bool = False,
    ) -> np.ndarray:
        """Solve GAM via PIRLS with line search stabilization.

        Args:
            max_iter: Maximum iterations.
            tol: Convergence tolerance on coefficient change.
            verbose: Print iteration progress.

        Returns:
            Fitted coefficient vector β.
        """
        for it in range(max_iter):
            # Compute predictions
            self.eta = self.X @ self.beta + self.offset
            self.mu = self.family.linkinv(self.eta)

            # Compute weights and working vector
            dmu_deta = self.family.dmu_deta(self.eta)
            var_mu = self.family.variance(self.mu, self.dispersion)
            
            # Handle zero/very small variances
            var_mu = np.maximum(var_mu, 1e-10)
            dmu_deta = np.where(np.abs(dmu_deta) < 1e-10, 1e-10, dmu_deta)
            
            # Weighted least squares: include observation weights
            # w_i = weights_i * (dμ/dη)² / Var(Y)
            w = self.weights * (dmu_deta**2) / var_mu
            
            # Working vector: z = η + weights * (y - μ) / dμ/dη
            z = self.eta + self.weights * (self.y - self.mu) / dmu_deta

            # Solve weighted least squares: (X^T W X + Sλ) β = X^T W z
            XtWX = self.X.T @ (self.X * w[:, np.newaxis])
            Xtwz = self.X.T @ (w * z)

            A = XtWX + self.S
            
            try:
                beta_new = linalg.solve(A, Xtwz)
            except linalg.LinAlgError:
                # Singular system: use least-squares solver
                beta_new = linalg.lstsq(A, Xtwz)[0]

            # 🔴 NEW: Line search for stability
            beta_new, step_size = self._line_search(
                self.beta, beta_new, self.eta
            )
            
            # Check for NaN/Inf
            if not np.all(np.isfinite(beta_new)):
                if verbose:
                    print(f"⚠️  Iteration {it}: Non-finite beta detected, reverting")
                beta_new = self.beta  # Use previous beta
                step_size = 0

            # 🔴 NEW: Improved convergence check (multiple criteria)
            dev = self._compute_deviance()
            delta_beta = np.max(np.abs(beta_new - self.beta))
            
            if len(self.dev_history) > 0:
                delta_dev = abs(dev - self.dev_history[-1])
            else:
                delta_dev = float('inf')
            
            # Track history
            obj = self._compute_objective()
            self.history.append({
                'iteration': it,
                'delta_beta': delta_beta,
                'objective': obj,
                'beta_norm': np.linalg.norm(beta_new),
                'step_size': step_size,
                'deviance': dev,
            })
            self.dev_history.append(dev)
            
            if verbose:
                print(
                    f'Iter {it:2d}: Δβ = {delta_beta:.6e}, '
                    f'ΔDev = {delta_dev:.6e}, step_size = {step_size:.2f}'
                )

            self.beta = beta_new

            # 🔴 NEW: Convergence check (all criteria must pass)
            if self._has_converged(delta_beta, delta_dev, tol):
                self.converged = True
                self.iterations = it + 1
                if verbose:
                    print(f'✓ Converged after {self.iterations} iterations')
                break

        if not self.converged and verbose:
            print(f'⚠️  Warning: Did not converge after {max_iter} iterations')

        self.iterations = it + 1
        return self.beta

    def _compute_objective(self) -> float:
        """Compute penalized deviance objective.

        L(β) = deviance + βᵀ Sλ β
        """
        deviance = self._compute_deviance()
        penalty = self.beta @ self.S @ self.beta
        return deviance + penalty

    def _compute_deviance(self) -> float:
        """Compute deviance (unpenalized)."""
        return -2 * self.family.loglik(self.y, self.mu, self.dispersion)

    def _line_search(self, beta_old: np.ndarray, beta_new: np.ndarray,
                     eta_old: np.ndarray, max_trials: int = 10) -> tuple:
        """Perform backtracking line search for step size.
        
        Returns:
            (beta_final, step_size) tuple
        """
        step_size = 1.0
        beta_direction = beta_new - beta_old
        
        # Store current state
        mu_old = self.mu.copy()
        beta_backup = self.beta.copy()
        dev_old = self._compute_deviance()
        
        for trial in range(max_trials):
            # Trial step
            beta_trial = beta_old + step_size * beta_direction
            eta_trial = self.X @ beta_trial + self.offset
            
            # Check if eta is in valid range
            eta_max = 100  # Prevent overflow in link functions
            if np.any(np.abs(eta_trial) > eta_max):
                step_size *= 0.5
                continue
            
            # Evaluate deviance
            try:
                mu_trial = self.family.linkinv(eta_trial)
                if np.any(mu_trial <= 0) or not np.all(np.isfinite(mu_trial)):
                    step_size *= 0.5
                    continue
                
                self.mu = mu_trial
                self.beta = beta_trial
                dev_trial = self._compute_deviance()
                
                # Accept if improvement (even small) or full step
                if dev_trial < dev_old * 0.99 or step_size > 0.9:
                    return beta_trial, step_size
                
                step_size *= 0.5
            except (ValueError, FloatingPointError):
                # Evaluation error, try smaller step
                step_size *= 0.5
                continue
        
        # Fallback: accept full Newton step even if small deviance increase
        # (this is better than reverting to old beta)
        self.mu = self.family.linkinv(self.X @ beta_new + self.offset)
        self.beta = beta_new
        return beta_new, 1.0

    def _has_converged(self, delta_beta: float, delta_dev: float,
                      tol: float) -> bool:
        """Check convergence via multiple criteria.
        
        All criteria must pass for convergence.
        """
        criteria = [
            ("beta_change", delta_beta < tol),
            ("deviance_change", delta_dev < tol),
            ("relative_change", delta_dev / (abs(self.dev_history[-1]) + 1e-10) < 1e-6),
        ]
        
        # All criteria must pass
        all_pass = all(c[1] for c in criteria)
        return all_pass

    def coefficients(self) -> np.ndarray:
        """Return fitted coefficients."""
        return self.beta

    def fitted_values(self) -> np.ndarray:
        """Return fitted mean values μ."""
        return self.family.linkinv(self.X @ self.beta + self.offset)

    def linear_predictor(self) -> np.ndarray:
        """Return linear predictor η = Xβ + offset."""
        return self.X @ self.beta + self.offset

    def residuals(self, type: str = 'deviance') -> np.ndarray:
        """Compute residuals.

        Args:
            type: One of 'deviance', 'pearson', 'response'.

        Returns:
            Residuals vector, shape (n,).
        """
        mu = self.fitted_values()
        
        if type == 'response':
            return self.y - mu
        elif type == 'pearson':
            var_mu = self.family.variance(mu, self.dispersion)
            return (self.y - mu) / np.sqrt(var_mu)
        elif type == 'deviance':
            # Deviance residuals (more complex, family-dependent)
            # For Gaussian: sqrt((y - mu)^2)
            # For Poisson/Gamma: sign(y - mu) * sqrt(...) based on loglik
            return np.sqrt(2 * (
                self.family.loglik(self.y, self.y, self.dispersion)
                - self.family.loglik(self.y, mu, self.dispersion)
            ))
        else:
            raise ValueError(f'Unknown residual type: {type}')

    def summary(self) -> str:
        """Return summary of fitting."""
        lines = [
            'PIRLS Solver',
            '============',
            f'Convergence: {self.converged}',
            f'Iterations: {self.iterations}',
            f'Final objective: {self.history[-1]["objective"]:.6e}',
            f'Final coefficient norm: {self.history[-1]["beta_norm"]:.6e}',
        ]
        return '\n'.join(lines)


def solve_pirls(
    X: np.ndarray,
    y: np.ndarray,
    family: object,
    S_list: list[np.ndarray],
    lambda_vec: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    dispersion: float = 1.0,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 25,
    tol: float = 1e-7,
) -> tuple[np.ndarray, bool]:
    """Functional API for PIRLS solver.

    Args:
        X: Design matrix.
        y: Response.
        family: Distribution family.
        S_list: Penalty matrices.
        lambda_vec: Smoothing parameters.
        offset: Offset.
        dispersion: Dispersion parameter.
        weights: Observation weights.
        max_iter: Max iterations.
        tol: Convergence tolerance.

    Returns:
        (beta, converged)
    """
    solver = PIRLSSolver(
        X, y, family, S_list,
        lambda_vec=lambda_vec,
        offset=offset,
        dispersion=dispersion,
        weights=weights,
    )
    beta = solver.solve(max_iter=max_iter, tol=tol, verbose=False)
    return beta, solver.converged
