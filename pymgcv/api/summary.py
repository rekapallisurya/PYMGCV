"""Model summary formatting in mgcv output format.

Returns summaries with:
- Parametric coefficients (Estimate, SE, t-value, p-value)
- Smooth term EDFs, F-stats, p-values
- Model statistics (deviance explained, AIC, REML, GCV)
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from pymgcv.api.gam import GAM
from pymgcv.diagnostics.significance_tests import SmoothTestSuite


def summary(model: GAM, detailed: bool = True) -> str:
    """Return mgcv-style model summary.

    Args:
        model: Fitted GAM model.
        detailed: Include smooth term details.

    Returns:
        Formatted summary string.

    Example:
        >>> from pymgcv.api.summary import summary
        >>> print(summary(model))
    """
    if not model.fitted:
        return '<Unfitted GAM>'

    lines = []

    # Header
    family_cls = model.family.__class__.__name__
    if hasattr(model.family, 'power'):
        family_cls += f'(p={model.family.power:.3f})'
    lines.append('Family: ' + family_cls)
    link_name = getattr(model.family, 'link', 'log')
    lines.append('Link function: ' + link_name)
    lines.append('')

    # Formula
    lines.append(f'Formula: {model.formula}')
    lines.append('')

    X = model.model_matrix.X
    y = model.model_matrix.response_vector()
    n = len(y)
    phi = getattr(model, 'dispersion_', 1.0)
    total_edf = getattr(model, 'edf', 1.0) or 1.0

    # ── Posterior precision A = X'WX + S_lambda  (used for both SE and Wald) ─
    from pymgcv.linalg.penalized_solver import PenalizedSolver
    offset = model.model_matrix.offset_vector()
    if offset is None:
        offset = np.zeros(n)
    eta = X @ model.beta + offset
    mu = model.family.linkinv(eta)
    dmu = model.family.dmu_deta(eta)
    # Use phi=1 for IRLS weights so that XtWX = X'W₁X (R convention).
    # Posterior covariance is Vb = phi * inv(X'W₁X + S), matching mgcv's Vp.
    var_mu = np.maximum(model.family.variance(mu, 1.0), 1e-10)
    w = np.clip(dmu ** 2 / var_mu, 1e-12, 1e8)
    XtWX = X.T @ (X * w[:, np.newaxis])
    S_combined = getattr(model, '_S_combined', np.zeros_like(XtWX))
    A = XtWX + S_combined                        # posterior precision (up to phi)
    try:
        from scipy import linalg as _la
        Ainv = _la.inv(A)                        # p×p; small since p≈design dim
    except Exception:
        Ainv = np.linalg.pinv(A)
    try:
        solver_vb = PenalizedSolver(XtWX, S_combined)
        se_all = np.sqrt(np.maximum(solver_vb.inv_diagonal() * phi, 0.0))
    except Exception:
        se_all = np.sqrt(np.maximum(np.diag(Ainv) * phi, 0.0))

    # ── Parametric coefficients ───────────────────────────────────────────────
    lines.append('Parametric coefficients:')
    lines.append('-' * 70)
    lines.append(f'{"":30s} {"Estimate":>12s} {"Std. Err":>10s} {"t value":>10s} {"Pr(>|t|)":>10s}')
    lines.append('-' * 70)

    pi = model.model_matrix.param_indices          # slice(0, n_param)
    param_slice = pi if pi is not None else slice(0, 1)
    col_names_all = model.model_matrix.column_names
    param_names = col_names_all[param_slice]  # list slice works for list
    beta_param = model.beta[param_slice]
    se_param = se_all[param_slice]
    dof_resid = max(n - total_edf, 1.0)

    for nm, coef, se in zip(param_names, beta_param, se_param):
        if se > 1e-12:
            t_val = coef / se
            p_val = float(2 * stats.t.sf(abs(t_val), df=dof_resid))
        else:
            t_val, p_val = 0.0, 1.0
        stars = ('***' if p_val < 0.001 else '**' if p_val < 0.01
                 else '*' if p_val < 0.05 else '.' if p_val < 0.1 else '')
        lines.append(
            f'{nm:30s} {coef:12.6f} {se:10.6f} {t_val:10.4f} {p_val:10.6f} {stars}'
        )
    lines.append('-' * 70)
    lines.append('')

    # ── Smooth term significance (Wood 2013 Wald test) ────────────────────────
    if detailed and model.edf_per_smooth:
        lines.append('Approximate significance of smooth terms:')
        lines.append('-' * 70)
        lines.append(f'{"":30s} {"edf":>7s} {"Ref.df":>8s} {"F":>10s} {"p-value":>10s}')
        lines.append('-' * 70)

        mm = model.model_matrix
        smooth_slices = list(mm.smooth_indices)
        smooth_label_list = list(model.edf_per_smooth.keys())

        for idx, label in enumerate(smooth_label_list):
            edf_info = model.edf_per_smooth[label]
            edf_val = edf_info['edf'] if isinstance(edf_info, dict) else float(edf_info)

            if idx < len(smooth_slices):
                sl = smooth_slices[idx]
                beta_j = model.beta[sl]
                Vb_j = Ainv[sl, sl] * phi        # block of full posterior cov
                try:
                    # Wald: T = beta_j' Vb_j^{-1} beta_j
                    T_j = float(beta_j @ np.linalg.solve(Vb_j, beta_j))
                except np.linalg.LinAlgError:
                    T_j = float(beta_j @ np.linalg.lstsq(Vb_j, beta_j, rcond=None)[0])
                F_j = T_j / max(edf_val, 1e-6)
                ref_df = round(edf_val + 0.01, 3)
                p_val = float(stats.f.sf(F_j, edf_val, dof_resid))
            else:
                F_j, ref_df, p_val = np.nan, edf_val, 1.0

            stars = ('***' if p_val < 0.001 else '**' if p_val < 0.01
                     else '*' if p_val < 0.05 else '.' if p_val < 0.1 else '')
            lines.append(
                f'{label:30s} {edf_val:7.3f} {ref_df:8.3f} {F_j:10.4f} {p_val:10.6f} {stars}'
            )

        lines.append('-' * 70)
        lines.append('')

    # ── Model statistics ──────────────────────────────────────────────────────
    lines.append('Model statistics:')
    lines.append('-' * 70)
    if total_edf:
        lines.append(f'Effective degrees of freedom: {total_edf:.2f}')

    try:
        fitted_mu = model.family.linkinv(X @ model.beta + offset)
        # Compute deviance using the unit deviance formula for each family.
        # For Tweedie we use the exact unit deviance (no Wright function, no phi):
        #   y>0: D_i = 2*[y*(y^{1-p}-μ^{1-p})/(1-p) - (y^{2-p}-μ^{2-p})/(2-p)]
        #   y=0: D_i = 2*μ^{2-p}/(2-p)
        # For other families we use -2*(loglik_fit - loglik_null) directly
        # (Wright-function terms cancel since both use the same y).
        from pymgcv.distributions.family_base import TweedieFamily as _TwFamS
        mu_null = np.full(n, float(np.maximum(y.mean(), 1e-10)))
        if isinstance(model.family, _TwFamS):
            _p = model.family.power
            def _tw_unit_dev(y_v: np.ndarray, mu_v: np.ndarray) -> float:
                """Tweedie deviance without Wright function or phi."""
                y_v  = np.asarray(y_v,  dtype=np.float64)
                mu_v = np.maximum(np.asarray(mu_v, dtype=np.float64), 1e-10)
                pos  = y_v > 0
                # Use max(y, 1) before raising to negative power to avoid 0**neg;
                # the np.where mask ensures the value is replaced by 0.0 anyway.
                y_safe = np.where(pos, y_v, 1.0)
                y_1mp = np.where(pos, y_safe ** (1 - _p), 0.0)
                y_2mp = np.where(pos, y_safe ** (2 - _p), 0.0)
                t1 = y_v * (y_1mp - mu_v ** (1 - _p)) / (1 - _p)
                t2 = (y_2mp - mu_v ** (2 - _p)) / (2 - _p)
                return float(2.0 * np.sum(t1 - t2))
            deviance = _tw_unit_dev(y, fitted_mu)
            null_dev = _tw_unit_dev(y, mu_null)
        else:
            # For Gaussian/Poisson/Gamma: use kernel difference (Wright terms absent).
            deviance = float(-2.0 * model.family.loglik(y, fitted_mu,  dispersion=1.0))
            null_dev = float(-2.0 * model.family.loglik(y, mu_null,    dispersion=1.0))
        dev_expl = (null_dev - deviance) / abs(null_dev) * 100 if null_dev != 0 else 0.0
        lines.append(f'Deviance explained: {dev_expl:.2f}%')
        aic_val = deviance + 2.0 * total_edf
        lines.append(f'AIC: {aic_val:.4f}')
        lines.append(f'Scale est. (dispersion phi): {phi:.6f}')
        lines.append(f'n: {n}')
    except Exception:
        pass

    lines.append('-' * 70)
    lines.append('')
    lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    lines.append('')

    return '\n'.join(lines)



class ModelSummary:
    """Enhanced summary object with individual statistics.

    Attributes:
        model: Fitted GAM.
        parametric_coefs: Table of parametric coefficients.
        smooth_stats: Table of smooth term statistics.
        model_stats: Model-level statistics dict.
    """

    def __init__(self, model: GAM) -> None:
        """Initialize summary.

        Args:
            model: Fitted GAM model.
        """
        self.model = model
        self._compute_stats()

    def _compute_stats(self) -> None:
        """Compute summary statistics."""
        # Parametric coefficients with standard errors
        try:
            X = self.model.model_matrix.X
            XtX_inv = np.linalg.inv(X.T @ X + 1e-8 * np.eye(X.shape[1]))
            se = np.sqrt(np.diag(XtX_inv))

            self.parametric_coefs = {
                'estimate': self.model.beta,
                'std_err': se,
                't_value': self.model.beta / (se + 1e-10),
                'p_value': 2 * (1 - stats.norm.cdf(np.abs(self.model.beta / (se + 1e-10)))),
            }
        except:
            self.parametric_coefs = None

        # Smooth statistics
        try:
            # Simplified: use EDF only
            self.smooth_stats = {
                'edf': getattr(self.model, 'edf_per_smooth', {}),
                'f_stat': [],
                'p_value': [],
            }
        except:
            self.smooth_stats = None

        # Model statistics
        self.model_stats = {
            'n_obs': len(self.model.data),
            'edf': self.model.edf,
            'deviance': self.model.deviance if hasattr(self.model, 'deviance') else None,
            'null_deviance': self.model.null_deviance if hasattr(self.model, 'null_deviance') else None,
            'aic': self.model.aic if hasattr(self.model, 'aic') else None,
        }

    def __str__(self) -> str:
        """Return formatted summary string."""
        return summary(self.model, detailed=True)
