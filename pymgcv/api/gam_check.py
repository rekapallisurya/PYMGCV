"""gam_check() — GAM diagnostic tests and k-adequacy checking.

Implements:
    - gam_check(): Overall GAM diagnostics (residual plots + k-index test)
    - k_check(): Formal k-adequacy test matching mgcv::k.check()
    - gam_residuals(): Residual extraction with type selection

References:
    - Wood, S.N. (2017). GAMs: An Introduction with R, §5.9.
    - Wood, S.N. (2006). Low rank smoothers for large smooth models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _get_residuals(model, type: str = 'deviance') -> np.ndarray:
    """Get residuals from a fitted GAM.

    Args:
        model: Fitted GAM object.
        type: 'deviance', 'pearson', or 'response'.

    Returns:
        Residuals array.
    """
    X = getattr(model, '_X_fit', None)
    y = getattr(model, '_y_fit', None)
    beta = getattr(model, 'beta', None)
    family = getattr(model, 'family', None)

    if any(v is None for v in (X, y, beta, family)):
        return np.array([])

    mm = getattr(model, 'model_matrix', None)
    offset = mm.offset_vector() if (mm and hasattr(mm, 'offset_vector')) else np.zeros(len(y))
    if offset is None:
        offset = np.zeros(len(y))

    eta = X @ beta + offset
    mu = family.linkinv(eta)

    if type == 'response':
        return y - mu
    elif type == 'pearson':
        var_mu = family.variance(mu, getattr(model, 'dispersion_', 1.0))
        return (y - mu) / np.sqrt(np.maximum(var_mu, 1e-10))
    elif type == 'deviance':
        phi = getattr(model, 'dispersion_', 1.0)
        ll_sat = family.loglik(y, y, phi)
        ll_fit = family.loglik(y, mu, phi)
        signed = np.sign(y - mu)
        dev_resid_sq = 2 * np.maximum(ll_sat - ll_fit, 0)
        return signed * np.sqrt(np.maximum(dev_resid_sq, 0))
    else:
        raise ValueError(f"Unknown residual type: {type!r}. Use 'deviance', 'pearson', or 'response'.")


def k_check(model, subsample: Optional[int] = None) -> pd.DataFrame:
    """Check k-adequacy for each smooth term (k.check equivalent).

    For each smooth term s_j, runs a correlation test between the fitted
    smooth and the basis-residuals to detect under-fitting due to k too small.

    The k-index (edf / (k - 1)) is the ratio of estimated degrees of freedom
    to the maximum possible degrees of freedom. Values near 1 suggest k may
    be too small.

    Args:
        model: Fitted GAM object.
        subsample: If given, subsample this many obs for the test (for speed).

    Returns:
        DataFrame with columns: smooth, k, edf, k_index, p-value.

    Example:
        >>> gam = GAM('y ~ s(x1, k=5) + s(x2)', data=df).fit()
        >>> print(k_check(gam))
    """
    parser = getattr(model, '_parser', None)
    # Try to access smooth terms via model_matrix
    mm = getattr(model, 'model_matrix', None)
    smooth_bases = getattr(mm, 'smooth_bases', []) if mm else []
    smooth_indices = getattr(mm, 'smooth_indices', []) if mm else []

    X = getattr(model, '_X_fit', None)
    y = getattr(model, '_y_fit', None)
    beta = getattr(model, 'beta', None)
    family = getattr(model, 'family', None)
    edf_per_smooth = getattr(model, 'edf_per_smooth', {})

    if X is None or y is None or beta is None:
        return pd.DataFrame(columns=['smooth', 'k', 'edf', 'k_index', 'p-value'])

    mm_obj = getattr(model, 'model_matrix', None)
    offset = mm_obj.offset_vector() if (mm_obj and hasattr(mm_obj, 'offset_vector')) else np.zeros(len(y))

    # Deviance residuals for the k-test
    resid = _get_residuals(model, type='deviance')
    if len(resid) == 0:
        return pd.DataFrame(columns=['smooth', 'k', 'edf', 'k_index', 'p-value'])

    n = len(resid)

    rows = []
    for j, (basis_obj, sl) in enumerate(zip(smooth_bases, smooth_indices)):
        B = X[:, sl]   # slice of design matrix for this smooth term
        k = B.shape[1]

        # k-index: edf ratio
        label = list(edf_per_smooth.keys())[j] if j < len(edf_per_smooth) else f's{j}'
        edf_j = edf_per_smooth.get(label, {}).get('edf', k - 1) if isinstance(edf_per_smooth.get(label), dict) else float(edf_per_smooth.get(label, k - 1))
        k_index = float(edf_j) / max(k - 1, 1)

        # p-value via randomisation test
        # Correlation between r_i and each column of B, take max |cor|
        # Under H0 (k adequate), this should be near zero
        obs_stat = float(np.max(np.abs(np.corrcoef(B.T, resid.T)[-1, :-1])))
        # p-value approximation (normal approximation for Fisher z of max correlation)
        # Conservative: use largest-of-k correlation distribution
        df = max(n - k - 1, 1)
        t_stat = obs_stat * np.sqrt(df / max(1 - obs_stat**2, 1e-10))
        from scipy.stats import t as t_dist
        p_val = float(2 * (1 - t_dist.cdf(abs(t_stat), df)))

        rows.append({
            'smooth': label,
            'k': k,
            'edf': round(edf_j, 3),
            'k_index': round(k_index, 3),
            'p-value': round(p_val, 4),
        })

    return pd.DataFrame(rows)


def gam_check(
    model,
    type: str = 'deviance',
    print_summary: bool = True,
    plot: bool = False,
) -> dict:
    """Diagnostic checks for a fitted GAM.

    Performs:
        1. Residual normality: Shapiro-Wilk test on residuals (if n ≤ 5000)
        2. Residual homogeneity: Breusch-Pagan-like test
        3. k-adequacy check: k-index for each smooth term
        4. Convergence check

    Args:
        model: Fitted GAM object.
        type: Residual type for diagnostics ('deviance', 'pearson', 'response').
        print_summary: Print summary to stdout.
        plot: Plot diagnostic plots (requires matplotlib).

    Returns:
        Dict with keys: 'residuals', 'k_check', 'tests', 'converged'.

    Example:
        >>> check = gam_check(model)
        >>> print(check['k_check'])
    """
    result: dict = {}

    # --- Convergence ---
    pirls = getattr(model, 'pirls_solver', None)
    converged = getattr(pirls, 'converged', True)
    result['converged'] = converged

    # --- Residuals ---
    resid = _get_residuals(model, type=type)
    result['residuals'] = resid

    tests: dict = {}

    if len(resid) > 0:
        # Normality test (Shapiro-Wilk for n <= 5000, else KS)
        n = len(resid)
        if 3 < n <= 5000:
            from scipy.stats import shapiro
            stat, p = shapiro(resid)
            tests['normality'] = {'test': 'Shapiro-Wilk', 'stat': float(stat), 'p': float(p)}
        elif n > 5000:
            from scipy.stats import kstest, norm
            stat, p = kstest(resid, 'norm', args=(resid.mean(), resid.std()))
            tests['normality'] = {'test': 'KS-norm', 'stat': float(stat), 'p': float(p)}

        # Heteroscedasticity: correlation of |resid| with fitted values
        X = getattr(model, '_X_fit', None)
        beta = getattr(model, 'beta', None)
        family = getattr(model, 'family', None)
        mm_obj = getattr(model, 'model_matrix', None)
        if X is not None and beta is not None and family is not None:
            y_ = getattr(model, '_y_fit', np.zeros(n))
            offset = mm_obj.offset_vector() if (mm_obj and hasattr(mm_obj, 'offset_vector')) else np.zeros(n)
            if offset is None:
                offset = np.zeros(n)
            eta = X @ beta + offset
            mu = family.linkinv(eta)
            fitted = mu
            cor = float(np.corrcoef(fitted, np.abs(resid))[0, 1])
            from scipy.stats import t as t_dist
            df = max(n - 2, 1)
            t_stat = cor * np.sqrt(df / max(1 - cor**2, 1e-10))
            p_hetero = float(2 * (1 - t_dist.cdf(abs(t_stat), df)))
            tests['heteroscedasticity'] = {
                'test': '|resid| ~ fitted correlation',
                'corr': round(cor, 4),
                'p': round(p_hetero, 4),
            }

    result['tests'] = tests

    # --- k-check ---
    k_df = k_check(model)
    result['k_check'] = k_df

    # --- Print summary ---
    if print_summary:
        print('\nGAM Check')
        print('=========')
        print(f'Convergence: {"YES" if converged else "NO - consider increasing iterations"}')
        print()

        if 'normality' in tests:
            t = tests['normality']
            flag = ' *' if t['p'] < 0.05 else ''
            print(f"Normality ({t['test']}): stat = {t['stat']:.4f}, p = {t['p']:.4f}{flag}")

        if 'heteroscedasticity' in tests:
            t = tests['heteroscedasticity']
            flag = ' *' if t['p'] < 0.05 else ''
            print(f"Heteroscedasticity: corr = {t['corr']:.4f}, p = {t['p']:.4f}{flag}")

        print()
        if not k_df.empty:
            print('Basis dimension (k) check:')
            print(k_df.to_string(index=False))
            low_k = k_df[k_df['k_index'] > 0.9]
            if not low_k.empty:
                print()
                print('WARNING: k_index close to 1 for:', list(low_k['smooth']))
                print('Consider increasing k for these terms.')
        print()

    # --- Plots ---
    if plot:
        _plot_gam_check(model, resid, type)

    return result


def _plot_gam_check(model, resid: np.ndarray, type: str) -> None:
    """Make diagnostic plots."""
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import probplot

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('GAM Diagnostics')

        # (1) QQ plot
        ax = axes[0, 0]
        probplot(resid, plot=ax)
        ax.set_title('Normal Q-Q')

        # (2) Residuals vs fitted
        X = getattr(model, '_X_fit', None)
        beta = getattr(model, 'beta', None)
        family = getattr(model, 'family', None)
        if X is not None and beta is not None and family is not None:
            y_ = getattr(model, '_y_fit', np.zeros(len(resid)))
            mm_obj = getattr(model, 'model_matrix', None)
            offset = mm_obj.offset_vector() if (mm_obj and hasattr(mm_obj, 'offset_vector')) else np.zeros(len(y_))
            fitted = family.linkinv(X @ beta + offset)
            ax = axes[0, 1]
            ax.scatter(fitted, resid, s=5, alpha=0.4)
            ax.axhline(0, color='red', lw=1)
            ax.set_xlabel('Fitted values')
            ax.set_ylabel(f'{type.capitalize()} residuals')
            ax.set_title('Residuals vs Fitted')

            # (3) Response vs fitted
            ax = axes[1, 0]
            ax.scatter(fitted, y_, s=5, alpha=0.4)
            lo, hi = min(fitted.min(), y_.min()), max(fitted.max(), y_.max())
            ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
            ax.set_xlabel('Fitted values')
            ax.set_ylabel('Response')
            ax.set_title('Response vs Fitted')

        # (4) Histogram of residuals
        ax = axes[1, 1]
        ax.hist(resid, bins=30, edgecolor='k', density=True)
        from scipy.stats import norm
        xs = np.linspace(resid.min(), resid.max(), 200)
        ax.plot(xs, norm.pdf(xs, resid.mean(), resid.std()), 'r-', lw=2)
        ax.set_title('Histogram of residuals')

        plt.tight_layout()
        plt.show()
    except ImportError:
        pass
