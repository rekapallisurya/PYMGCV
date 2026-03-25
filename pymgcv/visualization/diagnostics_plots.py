"""Diagnostic plots: QQ plots, residual vs fitted, etc."""

from __future__ import annotations

import numpy as np


def plot_model_diagnostics(model, axes=None):
    """Create a 2×2 panel of standard GAM diagnostic plots.

    Produces:
     - (0,0) QQ plot of deviance residuals
     - (0,1) Residuals vs linear predictor
     - (1,0) Histogram of residuals
     - (1,1) Response vs fitted values

    Args:
        model: Fitted GAM instance.
        axes: 2×2 array of matplotlib Axes, or None (creates figure).

    Returns:
        2×2 array of matplotlib Axes.
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        raise ImportError("matplotlib and scipy are required for diagnostic plots")

    if not model.fitted:
        raise ValueError("Model must be fitted before plotting")

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.tight_layout(pad=3.0)

    response_col = model.formula.split("~")[0].strip()
    y = model.data[response_col].values.astype(float)
    mu = model.predict(model.data, scale="response")
    eta = model.predict(model.data, scale="link")

    # Deviance residuals
    family = model.family

    def _dev_resids(y, mu):
        sign = np.sign(y - mu)
        try:
            ls = family.loglik(y, y)
            lf = family.loglik(y, mu)
            d = 2.0 * (ls - lf)
        except Exception:
            d = (y - mu) ** 2
        return sign * np.sqrt(np.abs(d))

    resids = _dev_resids(y, mu)

    # (0,0) QQ plot
    ax = axes[0, 0]
    (osm, osr), (slope, intercept, _) = stats.probplot(resids, fit=True)
    ax.scatter(osm, osr, s=8, alpha=0.6)
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1.5)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title("QQ plot of deviance residuals")

    # (0,1) Residuals vs linear predictor
    ax = axes[0, 1]
    ax.scatter(eta, resids, s=8, alpha=0.5)
    ax.axhline(0, color="red", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Linear predictor")
    ax.set_ylabel("Deviance residuals")
    ax.set_title("Residuals vs linear predictor")

    # (1,0) Histogram of residuals
    ax = axes[1, 0]
    ax.hist(resids, bins=30, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Deviance residuals")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of residuals")

    # (1,1) Response vs fitted
    ax = axes[1, 1]
    ax.scatter(mu, y, s=8, alpha=0.5)
    lo, hi = min(mu.min(), y.min()), max(mu.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "r-", linewidth=1.5)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Response")
    ax.set_title("Response vs fitted values")

    return axes
