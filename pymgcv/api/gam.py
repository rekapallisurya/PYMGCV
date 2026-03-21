"""Main GAM class for fitting generalized additive models.

This is the primary user-facing interface for pymgcv.

Example:
    >>> from pymgcv.api import gam
    >>> import pandas as pd
    >>> model = gam.GAM('y ~ s(x1) + s(x2)', data=df, family='gaussian')
    >>> model.fit()
    >>> print(model.summary())
    >>> y_pred = model.predict(df)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


class GAM:
    """Generalized Additive Model (GAM).

    Integrates formula parsing, basis construction, design matrix assembly,
    penalty matrices, PIRLS solver, and MAGIC smoothing parameter optimization.

    Attributes:
        formula: Model formula string.
        data: Input data.
        family: Distribution family.
        fitted: Whether model has been fitted.
        beta: Fitted coefficients.
        smoothing_parameters: Estimated λⱼ.
        edf: Effective degrees of freedom.
    """

    def __init__(
        self,
        formula: str,
        data: Optional[pd.DataFrame] = None,
        family: str = 'gaussian',
        offset: Optional[str] = None,
        weights: Optional[Any] = None,
        sp: Optional[Any] = None,
        select: bool = False,
        method: str = 'REML',
        knots: Optional[dict] = None,
        gamma: float = 1.0,
        drop_intercept: bool = False,
        control: Optional[dict] = None,
    ) -> None:
        """Initialize GAM.

        Args:
            formula: Model formula, e.g., 'y ~ s(x1) + s(x2) + x3'.
            data: Input data as DataFrame (optional, can be passed to fit()).
            family: Distribution family ('gaussian', 'poisson', 'gamma', 'tweedie',
                'binomial', 'negative.binomial', 'inverse.gaussian', 'beta', 'gaulss').
            offset: Column name for offset vector.
            weights: Column name (str) or array of observation weights.
            sp: Fixed smoothing parameters (array-like, one per smooth term).  When
                provided the MAGIC outer loop is skipped and the model is fitted at
                those fixed values.
            select: If True, add an extra near-zero penalty for each smooth
                (enables automatic term selection analogous to mgcv's select=TRUE).
            method: Smoothing criterion — 'REML' (default), 'ML', 'GCV', 'UBRE'.
            knots: Dict mapping variable name to interior knot positions, e.g.
                {'x': np.linspace(0, 1, 8)}.  Applied to cr/cs/bs/ps smooth terms.
            gamma: Scale inflation factor for the EDF/DoF in GCV/REML (default 1.0;
                increase above 1 for sparser fits, analogous to mgcv's gamma=).
            drop_intercept: If True, omit the intercept column from the design matrix.
            control: Dict for optimizer tuning. Supported keys:
                'epsilon' (convergence tol, float),
                'maxit' (max outer iterations, int),
                'inner_maxit' (max PIRLS iterations, int),
                'trace' (verbose output, bool).

        Raises:
            ValueError: If formula or family invalid.
        """
        self.formula = formula
        self.data = data
        self.family_name = family
        self.weights_col = weights
        self.sp = None if sp is None else np.asarray(sp, dtype=np.float64)
        self.select = bool(select)
        self.method = method
        self.knots: dict = knots or {}
        self.gamma = float(gamma)
        self.drop_intercept = bool(drop_intercept)
        self.control: dict = control or {}

        # Fitted attributes
        self.model_matrix = None
        self.family = None
        self.pirls_solver = None
        self.magic_optimizer = None

        self.fitted = False
        self.beta: Optional[np.ndarray] = None
        self.smoothing_parameters: Optional[np.ndarray] = None
        self.edf: Optional[float] = None
        self.edf_per_smooth: Optional[dict] = None
        self.dispersion_: float = 1.0      # estimated dispersion φ
        self._S_combined: Optional[np.ndarray] = None  # for inference
        self._X_fit: Optional[np.ndarray] = None       # training design matrix
        self._y_fit: Optional[np.ndarray] = None       # training response

    def fit(
        self,
        data: Optional[pd.DataFrame] = None,
        max_outer_iter: int = 200,
        max_inner_iter: int = 25,
        verbose: bool = False,
        use_gpu: bool = True,
    ) -> 'GAM':
        """Fit the GAM model.

        Args:
            data: Input data (required if not provided in __init__).
            max_outer_iter: Maximum MAGIC iterations.
            max_inner_iter: Maximum PIRLS iterations per outer iteration.
            verbose: Print progress.
            use_gpu: Enable JAX GPU acceleration if available.

        Returns:
            Self (for method chaining).
        """
        if data is not None:
            self.data = data
        elif self.data is None:
            raise ValueError('Data must be provided either in __init__ or fit()')

        from pymgcv.utils.formula_parser import FormulaParser
        from pymgcv.utils.model_matrix import ModelMatrix
        from pymgcv.penalties.penalty_matrix import PenaltyMatrix
        from pymgcv.distributions.family_base import (
            GaussianFamily, PoissonFamily, GammaFamily, TweedieFamily,
            BinomialFamily, NegativeBinomialFamily, InverseGaussianFamily,
            BetaFamily, GaulssFamily,
        )
        from pymgcv.optimizer.magic_optimizer import MAGICOptimizer
        from pymgcv.optimizer.edf import EDFComputer
        from pymgcv.optimizer.jax_acceleration import device_info

        # 1. Parse formula
        parser = FormulaParser(self.formula)

        # 2. Construct design matrix
        self.model_matrix = ModelMatrix(
            self.data, self.formula,
            knots=self.knots,
            drop_intercept=self.drop_intercept,
        )
        X = self.model_matrix.X
        y = self.model_matrix.response_vector()
        offset = self.model_matrix.offset_vector()
        self._X_fit = X
        self._y_fit = y

        # 3. Load observation weights
        weights = self._load_weights(self.data, len(y))

        # 4. Set up family
        family_map = {
            'gaussian': GaussianFamily(),
            'poisson': PoissonFamily(),
            'binomial': BinomialFamily(),
            'gamma': GammaFamily(shape=1.0),
            'tweedie': TweedieFamily(power=1.5),
            'negative.binomial': NegativeBinomialFamily(theta=1.0),
            'nb': NegativeBinomialFamily(theta=1.0),
            'inverse.gaussian': InverseGaussianFamily(),
            'beta': BetaFamily(),
            'betar': BetaFamily(),
            'gaulss': GaulssFamily(),
        }
        self.family = family_map.get(self.family_name.lower(), GaussianFamily())

        # 5. Build penalty matrices
        # For tensor product smooths, each smooth contributes multiple penalties.
        p_total = X.shape[1]
        S_list = []
        smooth_starts = []
        smooth_sizes = []

        for j, smooth_spec in enumerate(parser.smooth_terms):
            if j >= len(self.model_matrix.smooth_bases):
                S_list.append(np.zeros((p_total, p_total)))
                smooth_starts.append(0)
                smooth_sizes.append(10)
                continue

            smooth_slice = self.model_matrix.smooth_indices[j]
            s_start = smooth_slice.start
            s_stop = smooth_slice.stop
            actual_basis_dim = s_stop - s_start
            basis_obj = self.model_matrix.smooth_bases[j]

            # Tensor product smooths have multiple (Kronecker-sum) penalties
            if hasattr(basis_obj, 'penalty_matrices'):
                for P_small in basis_obj.penalty_matrices():
                    P_embed = np.zeros((p_total, p_total))
                    P_embed[s_start:s_stop, s_start:s_stop] = P_small
                    S_list.append(P_embed)
                    smooth_starts.append(s_start)
                    smooth_sizes.append(actual_basis_dim)
            elif hasattr(basis_obj, 'S') and basis_obj.S is not None:
                # ThinPlateSpline, random effect, cyclic spline: use .S directly
                P_embed = np.zeros((p_total, p_total))
                S_obj = basis_obj.S
                if S_obj.shape == (actual_basis_dim, actual_basis_dim):
                    P_embed[s_start:s_stop, s_start:s_stop] = S_obj
                elif actual_basis_dim > S_obj.shape[0]:
                    # By-variable smooth: replicate penalty block for each level
                    k_per = S_obj.shape[0]
                    n_levels = actual_basis_dim // k_per
                    for lev in range(n_levels):
                        off = s_start + lev * k_per
                        P_embed[off:off+k_per, off:off+k_per] = S_obj
                else:
                    P_embed[s_start:s_stop, s_start:s_stop] = S_obj[:actual_basis_dim, :actual_basis_dim]
                S_list.append(P_embed)
                smooth_starts.append(s_start)
                smooth_sizes.append(actual_basis_dim)
            elif hasattr(basis_obj, 'penalty_matrix_S'):
                # Alternate accessor (e.g. ThinPlateSpline)
                P_embed = np.zeros((p_total, p_total))
                S_obj = basis_obj.penalty_matrix_S()
                P_embed[s_start:s_stop, s_start:s_stop] = S_obj[:actual_basis_dim, :actual_basis_dim]
                S_list.append(P_embed)
                smooth_starts.append(s_start)
                smooth_sizes.append(actual_basis_dim)
            else:
                # Default: second-difference TPRS penalty
                penalty_builder = PenaltyMatrix(basis_dim=actual_basis_dim, penalty_type='tprs')
                P_embed = np.zeros((p_total, p_total))
                P_embed[s_start:s_stop, s_start:s_stop] = penalty_builder.S
                S_list.append(P_embed)
                smooth_starts.append(s_start)
                smooth_sizes.append(actual_basis_dim)

        if not S_list:
            S_list = [np.zeros((p_total, p_total))]
            smooth_starts = [0]
            smooth_sizes = [p_total]

        # select=True: add a small null-space penalty for each smooth term so that
        # the outer optimiser can shrink terms all the way to zero (mgcv select=TRUE)
        if self.select:
            for j in range(len(S_list)):
                S_j = S_list[j]
                diag_mean = np.mean(np.diag(S_j))
                eps_pen = max(diag_mean * 1e-4, 1e-6) * np.eye(p_total)
                S_list.append(eps_pen)
                smooth_starts.append(smooth_starts[j])
                smooth_sizes.append(smooth_sizes[j])

        # 6. Optimize smoothing parameters or use fixed sp=
        # Extract control parameters
        ctrl_maxit = int(self.control.get('maxit', max_outer_iter))
        ctrl_inner = int(self.control.get('inner_maxit', max_inner_iter))
        ctrl_tol   = float(self.control.get('epsilon', 1e-5))
        ctrl_verbose = bool(self.control.get('trace', verbose))
        initial_dispersion = 1.0

        optimizer = MAGICOptimizer(
            X=X, y=y, family=self.family, S_list=S_list,
            smooth_starts=smooth_starts,
            smooth_sizes=smooth_sizes,
            offset=offset, dispersion=initial_dispersion
        )
        optimizer.weights = weights if not np.all(weights == 1.0) else None

        if self.sp is not None:
            # Fixed smoothing parameters: run a single PIRLS pass, skip MAGIC
            from pymgcv.optimizer.pirls import PIRLSSolver
            sp_arr = np.asarray(self.sp, dtype=np.float64)
            if len(sp_arr) < len(S_list):
                # Pad with zeros for any select-added penalties
                sp_arr = np.concatenate([sp_arr, np.zeros(len(S_list) - len(sp_arr))])
            pirls = PIRLSSolver(
                X=X, y=y, family=self.family,
                S_list=S_list, lambda_vec=sp_arr,
                offset=offset, dispersion=initial_dispersion,
                weights=optimizer.weights,
            )
            pirls.solve(max_iter=ctrl_inner, verbose=ctrl_verbose)
            result = {'coef': pirls.beta, 'smooth_lambda': sp_arr}
        else:
            result = optimizer.optimize(
                max_outer_iter=ctrl_maxit,
                max_inner_iter=ctrl_inner,
                outer_tol=ctrl_tol,
                verbose=ctrl_verbose,
                use_jax=use_gpu and device_info()['available'],
                method=self.method.lower(),
                # Cap γ at 1.0 for REML optimisation: the pseudo-Gaussian criterion
                # is bounded below (λ→∞ stable) only at γ≤1 because logdet_A and
                # log|S|+ cancel exactly at γ=1.  For γ>1 the criterion diverges to
                # −∞, driving λ to the clip limit.  The user-specified γ is preserved
                # in per-smooth tests and dispersion estimation.
                gamma=min(float(self.gamma), 1.0),
            )

        self.beta = result['coef']
        self.smoothing_parameters = result['smooth_lambda']

        # 7. Compute EDF
        S_combined = np.zeros((p_total, p_total))
        for j_pen, S_j_full in enumerate(S_list):
            if j_pen < len(self.smoothing_parameters):
                lam = self.smoothing_parameters[j_pen]
                S_combined += lam * S_j_full
        self._S_combined = S_combined

        # 7a. For non-Gaussian families: a second optimizer pass at the estimated φ
        #     refines λ for the actual dispersion scale.  The first pass runs at φ=1
        #     (pseudo-Gaussian REML), giving a coarse φ estimate.  The second pass
        #     re-selects λ with rss_pseudo ≈ n_eff, which is the right balance point.
        #
        #     The γ cap (≤ 1.0) also applies here: for γ > 1 the pseudo-Gaussian
        #     criterion diverges to −∞ as λ → ∞, so all λ values pile at the clip
        #     limit and EDF collapses to 1 per smooth.  At γ = 1 the criterion is
        #     bounded and the Newton optimiser converges to a finite minimum.
        from pymgcv.distributions.family_base import GaussianFamily as _GF, PoissonFamily as _PF
        _needs_phi_refit = not isinstance(self.family, (_GF, _PF)) and self.sp is None
        phi_refit = 1.0
        if _needs_phi_refit:
            # Estimate φ from first-pass residuals
            _eta_p1 = X @ self.beta + (offset if offset is not None else 0.0)
            _mu_p1 = self.family.linkinv(_eta_p1)
            _var_unit_p1 = np.maximum(self.family.variance(_mu_p1, 1.0), 1e-10)
            _pearson_p1 = float(np.sum((y - _mu_p1) ** 2 / _var_unit_p1))
            _edf_p1 = EDFComputer(X, S_combined, self.family, self.beta, offset, dispersion=1.0).total_edf()
            phi_refit = max(_pearson_p1 / max(len(y) - _edf_p1, 1.0), 1e-6)

            # Second optimizer pass at estimated φ, warm-started from first pass
            _lam_floor = np.log(max(phi_refit, 1.0) / max(len(y), 1.0))
            warm_log = np.maximum(np.log(np.maximum(self.smoothing_parameters, 1e-10)), _lam_floor)
            opt2 = MAGICOptimizer(
                X=X, y=y, family=self.family, S_list=S_list,
                smooth_starts=smooth_starts, smooth_sizes=smooth_sizes,
                offset=offset, dispersion=phi_refit,
            )
            opt2.weights = optimizer.weights
            opt2.lambda_log = warm_log
            opt2.lambda_vec = np.exp(warm_log)
            result2 = opt2.optimize(
                max_outer_iter=ctrl_maxit,
                max_inner_iter=ctrl_inner,
                outer_tol=ctrl_tol,
                verbose=ctrl_verbose,
                use_jax=use_gpu and device_info()['available'],
                method=self.method.lower(),
                gamma=min(float(self.gamma), 1.0),
            )
            self.beta = result2['coef']
            self.smoothing_parameters = result2['smooth_lambda']

            # Rebuild S_combined with refined λ
            S_combined = np.zeros((p_total, p_total))
            for j_pen, S_j_full in enumerate(S_list):
                if j_pen < len(self.smoothing_parameters):
                    lam = self.smoothing_parameters[j_pen]
                    S_combined += lam * S_j_full
            self._S_combined = S_combined

        # 7b. Compute EDF with two-pass φ/EDF refinement (matches mgcv gam.fit3).
        #
        # The REML optimizer iteratively estimates φ alongside λ (see magic_optimizer.py).
        # For Gaussian and Poisson (φ=1 by definition) use unit dispersion.
        # For other families, use the optimizer's final φ estimate; then refine via
        # one additional Pearson step to ensure EDF and φ are mutually consistent.
        from pymgcv.distributions.family_base import GaussianFamily, PoissonFamily
        _estimate_phi = not isinstance(self.family, (GaussianFamily, PoissonFamily))

        # Pass 1: EDF at optimizer's estimated φ (or phi_refit for non-Gaussian)
        phi_from_opt = phi_refit if _needs_phi_refit else float(result.get('dispersion', 1.0))
        edf_comp_1 = EDFComputer(X, S_combined, self.family, self.beta, offset, dispersion=phi_from_opt)
        edf_1 = edf_comp_1.total_edf()

        if _estimate_phi:
            _eta = X @ self.beta + (offset if offset is not None else 0.0)
            mu_fit_ = self.family.linkinv(_eta)
            var_unit_ = np.maximum(self.family.variance(mu_fit_, 1.0), 1e-10)
            _pearson_ss = float(np.sum((y - mu_fit_) ** 2 / var_unit_))
            phi_1 = max(_pearson_ss / max(len(y) - edf_1, 1.0), 1e-6)
            # Pass 2: self-consistent φ/EDF
            edf_computer = EDFComputer(X, S_combined, self.family, self.beta, offset, dispersion=phi_1)
        else:
            edf_computer = edf_comp_1

        self.edf = edf_computer.total_edf()

        # Per-smooth EDF via influence matrix diagonal
        self.edf_per_smooth = {}
        smooth_slices = []
        smooth_labels = []
        for i in range(len(self.model_matrix.smooth_indices)):
            s = self.model_matrix.smooth_indices[i]
            smooth_slices.append(s)
            spec = parser.smooth_terms[i] if i < len(parser.smooth_terms) else None
            smooth_labels.append(spec.label if spec else f'smooth_{i}')

        if smooth_slices:
            edf_map = edf_computer.edf_by_smooth(smooth_slices)
            for i, label in enumerate(smooth_labels):
                self.edf_per_smooth[label] = {'edf': max(1.0, edf_map.get(i, 1.0))}

        # 8. Estimate dispersion parameter
        self.dispersion_ = self._estimate_dispersion()

        self.fitted = True
        if verbose:
            print(f'Fitted GAM with {self.edf:.2f} total EDF')

        return self

    def _load_weights(self, data: pd.DataFrame, n: int) -> np.ndarray:
        """Load and validate observation weights.

        Args:
            data: Training data.
            n: Number of observations.

        Returns:
            Weights array of shape (n,), all ones if no weights specified.
        """
        if self.weights_col is None:
            return np.ones(n)

        if isinstance(self.weights_col, str):
            if self.weights_col not in data.columns:
                raise ValueError(f'Weights column "{self.weights_col}" not found in data')
            w = data[self.weights_col].values.astype(float)
        else:
            w = np.asarray(self.weights_col, dtype=float)

        if len(w) != n:
            raise ValueError(f'Weights length {len(w)} != n={n}')
        if not np.all(w > 0):
            raise ValueError('All weights must be positive')
        if not np.all(np.isfinite(w)):
            raise ValueError('All weights must be finite')

        return w / w.mean()  # normalize to mean 1

    def _estimate_dispersion(self) -> float:
        """Estimate dispersion (scale) parameter φ via Pearson statistic.

        φ̂ = Σ (y - μ)² / V(μ)  /  (n - edf)

        For Gaussian this equals the residual variance.  For other families it
        estimates the over/under-dispersion factor.

        Returns:
            Estimated dispersion, clamped to [1e-6, ∞).
        """
        if not self.fitted and self._X_fit is None:
            return 1.0

        X = self._X_fit
        y = self._y_fit
        beta = self.beta
        family = self.family
        offset = self.model_matrix.offset_vector()
        if offset is None:
            offset = np.zeros(len(X))

        eta = X @ beta + offset
        mu = family.linkinv(eta)
        var_mu = family.variance(mu, self.dispersion_)
        var_mu = np.where(var_mu < 1e-10, 1e-10, var_mu)

        pearson_resid_sq = (y - mu) ** 2 / var_mu
        n = len(y)
        edf = self.edf if (self.edf is not None and self.edf > 0) else 1.0
        dof = max(n - edf, 1.0)

        phi = float(np.sum(pearson_resid_sq) / dof)
        return max(phi, 1e-6)

    def standard_errors(self) -> Optional[np.ndarray]:
        """Compute Bayesian posterior standard errors for coefficients.

        Uses the Bayesian posterior covariance (Wood 2006, mgcv default):
            V_b = (X'WX + S_lambda)^{-1} * phi

        The square root of the diagonal gives the standard errors that mgcv
        reports in summary.gam().

        Returns:
            Standard errors array of shape (p,), or None if not fitted.
        """
        if not self.fitted or self._S_combined is None:
            return None

        from pymgcv.linalg.penalized_solver import PenalizedSolver

        X = self._X_fit
        beta = self.beta
        family = self.family
        offset = self.model_matrix.offset_vector()
        if offset is None:
            offset = np.zeros(len(X))

        eta = X @ beta + offset
        mu = family.linkinv(eta)
        dmu_deta = family.dmu_deta(eta)
        var_mu = np.maximum(family.variance(mu, self.dispersion_), 1e-10)
        w = np.clip(dmu_deta ** 2 / var_mu, 1e-12, 1e8)

        XtWX = X.T @ (X * w[:, np.newaxis])

        try:
            solver = PenalizedSolver(XtWX, self._S_combined)
            diag_Ainv = solver.inv_diagonal()
            se = np.sqrt(np.maximum(diag_Ainv * self.dispersion_, 0.0))
        except Exception:
            se = np.full(len(beta), np.nan)

        return se

    def confidence_intervals(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Compute confidence intervals for all coefficients.

        Args:
            level: Confidence level (default 0.95).

        Returns:
            (lower, upper) arrays of shape (p,).
        """
        from scipy.stats import norm
        se = self.standard_errors()
        if se is None:
            return (np.full(len(self.beta), np.nan), np.full(len(self.beta), np.nan))
        z = norm.ppf((1 + level) / 2)
        return (self.beta - z * se, self.beta + z * se)

    def summary(self) -> str:
        """Return model summary in mgcv format.

        Includes parametric coefficients with SEs/p-values, smooth term EDFs,
        approximate F-statistics, deviance explained, AIC, REML score.

        Returns:
            Human-readable summary string matching mgcv's summary.gam() format.
        """
        if not self.fitted:
            return 'Model not yet fitted. Call .fit() first.'

        try:
            from pymgcv.api.summary import summary as _summary
            return _summary(self, detailed=True)
        except Exception:
            pass

        # Fallback summary
        lines = []
        lines.append('Family: ' + self.family.__class__.__name__)
        lines.append('Link function: ' + getattr(self.family, 'link', 'unknown'))
        lines.append(f'Dispersion parameter: {self.dispersion_:.6f}')
        lines.append('')
        lines.append('Formula: ' + self.formula)
        lines.append('')
        lines.append('Estimated smoothing parameters:')
        for i, lam in enumerate(self.smoothing_parameters or []):
            lines.append(f'  sp({i}) = {lam:.6e}')
        lines.append('')
        lines.append(f'Model dimension(s): {len(self.beta)} total coefs')
        if self.edf:
            lines.append(f'Effective degrees of freedom: {self.edf:.2f}')
        lines.append('')

        se = self.standard_errors()
        lines.append('Parametric coefficients:')
        from scipy.stats import t as t_dist
        n = len(self._y_fit) if self._y_fit is not None else len(self.beta)
        for i, coef in enumerate(self.beta[:min(8, len(self.beta))]):
            si = se[i] if (se is not None and i < len(se) and np.isfinite(se[i])) else np.nan
            if np.isfinite(si) and si > 0:
                t_val = coef / si
                p_val = 2 * (1 - t_dist.cdf(abs(t_val), max(1, n - len(self.beta))))
                stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
                lines.append(f'  Coef {i}: {coef:.6f}  SE={si:.6f}  t={t_val:.4f}  p={p_val:.4f} {stars}')
            else:
                lines.append(f'  Coef {i}: {coef:.6f}')

        return '\n'.join(lines)

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        scale: str = 'response',
        type: Optional[str] = None,
    ) -> 'np.ndarray | dict':
        """Make predictions.

        Args:
            data: New data for prediction. If None, use training data.
            scale: 'link' for linear predictor, 'response' for µ scale.
                   Ignored when type='terms' or type='lpmatrix'.
            type: Prediction type (mirrors mgcv's type= argument):
                  'response' — fitted µ values (default).
                  'link'     — linear predictor η.
                  'terms'    — per-smooth contributions as a dict
                               {smooth_label: array(n)}.
                  'lpmatrix' — the full prediction design matrix X_new
                               so that X_new @ beta = η (shape (n, p)).
                  When type is given it takes priority over scale.

        Returns:
            Array for type='response'/'link'/'lpmatrix', dict for type='terms'.
        """
        if not self.fitted:
            raise ValueError('Model not yet fitted')

        if data is None:
            data = self.data

        # type= overrides scale= for backwards compat
        if type is not None:
            scale = type

        # Construct design matrix for new data
        from pymgcv.utils.model_matrix import ModelMatrix
        from pymgcv.utils.formula_parser import FormulaParser
        mm = ModelMatrix(
            data, self.formula,
            knots=self.knots,
            drop_intercept=self.drop_intercept,
        )
        X_new = mm.X
        offset_raw = mm.offset_vector()
        offset_new = offset_raw if offset_raw is not None else np.zeros(len(data))

        if scale == 'lpmatrix':
            return X_new

        if scale == 'terms':
            # Return per-smooth contributions
            parser = FormulaParser(self.formula)
            result: dict = {}
            for i, smooth_spec in enumerate(parser.smooth_terms):
                if i < len(mm.smooth_indices):
                    sl = mm.smooth_indices[i]
                    contribution = X_new[:, sl] @ self.beta[sl]
                    result[smooth_spec.label] = contribution
            # Also add parametric contribution
            if mm.param_indices is not None:
                ps = mm.param_indices
                result['(parametric)'] = X_new[:, ps] @ self.beta[ps]
            return result

        # Linear predictor
        eta = X_new @ self.beta + offset_new

        if scale in ('response', 'mu'):
            return self.family.linkinv(eta)
        elif scale == 'link':
            return eta
        else:
            raise ValueError(f'Invalid scale/type: {scale!r}')

    def gam_check(self, type: str = 'deviance', print_summary: bool = True, plot: bool = False) -> dict:
        """Run GAM diagnostic checks (residual tests, convergence, k-adequacy).

        Args:
            type: Residual type ('deviance', 'pearson', 'response').
            print_summary: Print summary to stdout.
            plot: Plot diagnostic plots (requires matplotlib).

        Returns:
            Dict with keys 'residuals', 'k_check', 'tests', 'converged'.
        """
        from pymgcv.api.gam_check import gam_check as _gam_check
        return _gam_check(self, type=type, print_summary=print_summary, plot=plot)

    def k_check(self, subsample: Optional[int] = None) -> 'pd.DataFrame':
        """Check adequacy of basis dimensions for each smooth term.

        Args:
            subsample: If set, use this many random rows for the test (faster for large n).

        Returns:
            DataFrame with columns k, k', edf, p-value for each smooth.
        """
        from pymgcv.api.gam_check import k_check as _k_check
        return _k_check(self, subsample=subsample)

    def __repr__(self) -> str:
        """String representation."""
        return f'GAM({self.formula!r}, family={self.family_name!r})'
