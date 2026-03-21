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

        # Family-specific mu initialization
        self._initialize_from_family()

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

    def _initialize_from_family(self) -> None:
        """Set sensible starting beta/mu/eta using family.initialize()."""
        if not hasattr(self.family, 'initialize'):
            return  # Use defaults (zeros)

        mu_init = self.family.initialize(self.y)
        mu_init = np.asarray(mu_init, dtype=np.float64)
        self.mu = mu_init

        if hasattr(self.family, 'linkfun'):
            eta_init = self.family.linkfun(mu_init)
        else:
            # Fallback: try family.linkinv inverse numerically (not used)
            eta_init = np.zeros_like(mu_init)

        # If there's an intercept (first column all-ones), set beta[0] so that
        # X @ beta + offset ≈ mean(eta_init) for the intercept-only model.
        if self.p > 0 and np.allclose(self.X[:, 0], 1.0):
            target_eta = np.mean(eta_init - self.offset)
            if np.isfinite(target_eta):
                self.beta[0] = target_eta

        # Recompute eta and mu from initialized beta
        self.eta = self.X @ self.beta + self.offset
        self.mu = self.family.linkinv(self.eta)

        # Clip mu to valid domain to avoid numerical issues
        self._clip_mu()

    def _clip_mu(self) -> None:
        """Clip mu to the valid domain for the current family."""
        family_name = type(self.family).__name__.lower()
        if 'binomial' in family_name:
            self.mu = np.clip(self.mu, 1e-6, 1.0 - 1e-6)
        elif any(x in family_name for x in ('poisson', 'gamma', 'tweedie', 'negative', 'inverse')):
            self.mu = np.maximum(self.mu, 1e-6)

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
        beta_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve GAM via PIRLS.

        Uses the PenalizedSolver (Cholesky → ridge → SVD) for the inner
        linear system.  Convergence criterion matches mgcv exactly:

            |dev_old - dev_new| / (0.1 + |dev_new|) < tol

        Step-halving prevents deviance from increasing.

        Args:
            max_iter: Maximum PIRLS iterations.
            tol: Convergence tolerance (deviance change, relative).
            verbose: Print per-iteration progress.
            beta_init: Warm-start coefficients (optional); if None, uses the
                       family-initialised beta from __init__.

        Returns:
            Fitted coefficient vector β.
        """
        from pymgcv.linalg.penalized_solver import PenalizedSolver

        if beta_init is not None:
            self.beta = np.asarray(beta_init, dtype=np.float64).copy()
            # Recompute eta/mu from warm-start beta
            self.eta = self.X @ self.beta + self.offset
            self.mu = self.family.linkinv(self.eta)
            self._clip_mu()

        dev_old = np.inf
        last_solver: Optional[PenalizedSolver] = None
        last_XtWX: Optional[np.ndarray] = None

        for it in range(max_iter):
            # ----------------------------------------------------------
            # 1. Linear predictor and mean at current beta
            # ----------------------------------------------------------
            self.eta = self.X @ self.beta + self.offset
            self.mu = self.family.linkinv(self.eta)
            self._clip_mu()

            # ----------------------------------------------------------
            # 2. IRLS weights and working response
            # ----------------------------------------------------------
            dmu_deta = self.family.dmu_deta(self.eta)
            var_mu = self.family.variance(self.mu, self.dispersion)

            var_mu = np.maximum(var_mu, 1e-10)
            # Safe dmu/deta: avoid division by zero in working response
            dmu_safe = np.where(np.abs(dmu_deta) < 1e-10,
                                np.sign(dmu_deta + 1e-30) * 1e-10,
                                dmu_deta)

            # IRLS weights (include observation weights)
            w = self.weights * np.clip((dmu_safe ** 2) / var_mu, 1e-12, 1e8)

            # Working response z = eta + (y - mu) / dmu_deta
            # Remove offset so the system solves for β only
            z_adj = self.eta + (self.y - self.mu) / dmu_safe - self.offset

            # ----------------------------------------------------------
            # 3. Penalized WLS system: (X'WX + S) beta_new = X'W z_adj
            # ----------------------------------------------------------
            XtWX = self.X.T @ (self.X * w[:, np.newaxis])
            Xtwz = self.X.T @ (w * z_adj)

            last_XtWX = XtWX
            last_solver = PenalizedSolver(XtWX, self.S)
            beta_cand = last_solver.solve(Xtwz)

            if not np.all(np.isfinite(beta_cand)):
                # Solver failed: stay at current beta and exit
                if verbose:
                    print(f'  PIRLS iter {it}: solver returned non-finite beta')
                break

            # ----------------------------------------------------------
            # 4. Step-halving: ensure deviance does not increase
            # ----------------------------------------------------------
            beta_cand = self._step_halve(self.beta, beta_cand)
            self.beta = beta_cand

            # ----------------------------------------------------------
            # 5. Penalized deviance at updated beta (matches mgcv's pdev)
            # ----------------------------------------------------------
            self.eta = self.X @ self.beta + self.offset
            self.mu = self.family.linkinv(self.eta)
            self._clip_mu()
            dev_new = self._compute_penalized_deviance()

            # Track per-iteration history (step_size=1 when no halving happened)
            self.dev_history.append(dev_new)
            self.history.append({'step_size': 1.0, 'deviance': dev_new,
                                  'iteration': it})

            if verbose:
                rel = abs(dev_old - dev_new) / (0.1 + abs(dev_new))
                print(f'  PIRLS iter {it:2d}: pdev={dev_new:.6f}  '
                      f'Δpdev/pdev={rel:.2e}')

            # ----------------------------------------------------------
            # 6. mgcv-style convergence check on penalized deviance
            # ----------------------------------------------------------
            if abs(dev_old - dev_new) / (0.1 + abs(dev_new)) < tol:
                self.converged = True
                self.iterations = it + 1
                break

            dev_old = dev_new

        else:
            self.iterations = max_iter

        # ----------------------------------------------------------
        # 7. Rebuild solver at final beta for covariance / EDF reuse
        # ----------------------------------------------------------
        self.eta = self.X @ self.beta + self.offset
        self.mu = self.family.linkinv(self.eta)
        self._clip_mu()

        dmu = self.family.dmu_deta(self.eta)
        vm = np.maximum(self.family.variance(self.mu, self.dispersion), 1e-10)
        dmu_s = np.where(np.abs(dmu) < 1e-10, 1e-10, dmu)
        w_final = self.weights * np.clip((dmu_s ** 2) / vm, 1e-12, 1e8)

        self.last_XtWX_ = self.X.T @ (self.X * w_final[:, np.newaxis])
        self.last_solver_ = PenalizedSolver(self.last_XtWX_, self.S)

        return self.beta

    def _step_halve(
        self,
        beta_old: np.ndarray,
        beta_cand: np.ndarray,
        max_halvings: int = 25,
    ) -> np.ndarray:
        """Halve the step if deviance increases (mgcv-style safeguard).

        Accepts the candidate if deviance does not increase by more than
        a tiny tolerance (0.0001 %), matching mgcv's step.failed criterion.
        """
        direction = beta_cand - beta_old

        # Penalized deviance at old beta: -2*loglik + beta'S*beta / phi
        # (matches mgcv's pdev criterion; penalty term ensures we accept smoother steps)
        pdev_curr = (-2.0 * self.family.loglik(self.y, self.mu, self.dispersion)
                     + float(beta_old @ self.S @ beta_old) / self.dispersion)

        step = 1.0
        for _ in range(max_halvings):
            beta_try = beta_old + step * direction
            eta_try = self.X @ beta_try + self.offset

            # Prevent exp-overflow in exponential families
            if np.max(np.abs(eta_try)) > 100:
                step *= 0.5
                continue

            try:
                mu_try = self.family.linkinv(eta_try)
            except Exception:
                step *= 0.5
                continue

            if not np.all(np.isfinite(mu_try)):
                step *= 0.5
                continue

            # Clip mu_try for the deviance evaluation
            family_name = type(self.family).__name__.lower()
            if 'binomial' in family_name:
                mu_try = np.clip(mu_try, 1e-6, 1.0 - 1e-6)
            elif any(nm in family_name for nm in ('poisson', 'gamma', 'tweedie', 'negative', 'inverse')):
                mu_try = np.maximum(mu_try, 1e-6)

            # Penalized deviance at beta_try
            pdev_try = (-2.0 * self.family.loglik(self.y, mu_try, self.dispersion)
                        + float(beta_try @ self.S @ beta_try) / self.dispersion)

            # Accept if penalized deviance didn't significantly increase
            # (matches mgcv: step.failed when pdev > pdev_old * 1.0000001 + eps)
            if pdev_try <= pdev_curr + abs(pdev_curr) * 1e-7 + 1e-10:
                return beta_try

            step *= 0.5

        # No improvement found: return old beta (do not move)
        return beta_old

    def _compute_deviance(self) -> float:
        """Compute unpenalized deviance -2 * loglik(y, mu, phi) at current mu."""
        return -2.0 * self.family.loglik(self.y, self.mu, self.dispersion)

    def _compute_penalized_deviance(self) -> float:
        """Compute penalized deviance = -2*loglik + beta'S*beta/phi.

        This matches mgcv's pdev convergence criterion, which is critical for
        correct PIRLS behaviour when warm-starting from a different lambda.
        """
        unpen = -2.0 * self.family.loglik(self.y, self.mu, self.dispersion)
        pen = float(self.beta @ self.S @ self.beta) / self.dispersion
        return unpen + pen

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
        """Compute residuals of type 'deviance', 'pearson', or 'response'."""
        mu = self.fitted_values()
        if type == 'response':
            return self.y - mu
        elif type == 'pearson':
            var_mu = self.family.variance(mu, self.dispersion)
            return (self.y - mu) / np.sqrt(np.maximum(var_mu, 1e-10))
        elif type == 'deviance':
            ll_sat = self.family.loglik(self.y, self.y, self.dispersion)
            ll_fit = self.family.loglik(self.y, mu, self.dispersion)
            return np.sign(self.y - mu) * np.sqrt(np.maximum(2.0 * (ll_sat - ll_fit), 0.0))
        else:
            raise ValueError(f'Unknown residual type: {type}')

    def summary(self) -> str:
        """Brief summary string."""
        return (f'PIRLSSolver: converged={self.converged}, '
                f'iterations={self.iterations}')


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
    """Functional API for PIRLS solver."""
    solver = PIRLSSolver(
        X, y, family, S_list,
        lambda_vec=lambda_vec,
        offset=offset,
        dispersion=dispersion,
        weights=weights,
    )
    beta = solver.solve(max_iter=max_iter, tol=tol, verbose=False)
    return beta, solver.converged
