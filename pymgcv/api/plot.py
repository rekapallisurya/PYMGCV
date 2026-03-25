"""Visualization for GAM models.

Provides plotting functions:
    - plot_smooth: 1D smooth term effects
    - plot_smooth_2d: 2D tensor product smooth
    - plot_residuals: Residual diagnostics
    - plot_diagnostics: 4-panel diagnostic plot
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from pymgcv.api.gam import GAM
from pymgcv.api.predict import Predictor


def plot_smooth(
    model: GAM,
    var_name: str,
    ax: plt.Axes | None = None,
    n_grid: int = 100,
    confidence_band: bool = True,
    ci: float = 0.95,
) -> plt.Axes:
    """Plot 1D smooth term effects.

    Args:
        model: Fitted GAM.
        var_name: Variable name.
        ax: Matplotlib axes (default: create new).
        n_grid: Grid size for smooth curve.
        confidence_band: Include confidence bands.
        ci: Confidence interval level.

    Returns:
        Matplotlib axes.

    Example:
        >>> from pymgcv.api.plot import plot_smooth
        >>> fig, ax = plt.subplots()
        >>> plot_smooth(model, 'x1', ax=ax)
        >>> plt.show()
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib not installed. Install with: pip install matplotlib")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if var_name not in model.data.columns:
        raise ValueError(f"{var_name} not in data")

    # Get data range
    x_vals = model.data[var_name].values
    x_min, x_max = np.percentile(x_vals, [1, 99])
    x_grid = np.linspace(x_min, x_max, n_grid)

    # Create grid data
    grid_data = model.data.iloc[[0] * n_grid].copy()
    grid_data[var_name] = x_grid

    # Predict
    predictor = Predictor(model)
    pred_df = predictor.predict(grid_data, scale="response", ci=ci)

    # Plot
    ax.plot(x_grid, pred_df["fit"], "b-", linewidth=1.5, label="Fitted smooth")

    if confidence_band:
        ax.fill_between(
            x_grid,
            pred_df["lwr"],
            pred_df["upr"],
            alpha=0.3,
            color="blue",
            label=f"{int(ci*100)}% CI",
        )

    ax.set_xlabel(var_name)
    ax.set_ylabel("Effect")
    ax.set_title(f"Smooth term: {var_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_smooth_2d(
    model: GAM,
    var1: str,
    var2: str,
    ax: plt.Axes | None = None,
    n_grid: int = 50,
    cmap: str = "viridis",
) -> plt.Axes:
    """Plot 2D tensor product smooth.

    Args:
        model: Fitted GAM.
        var1: First variable name.
        var2: Second variable name.
        ax: Matplotlib axes.
        n_grid: Grid size.
        cmap: Colormap.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    # Create 2D grid
    x1_vals = model.data[var1].values
    x2_vals = model.data[var2].values
    x1_min, x1_max = np.percentile(x1_vals, [1, 99])
    x2_min, x2_max = np.percentile(x2_vals, [1, 99])

    x1_grid = np.linspace(x1_min, x1_max, n_grid)
    x2_grid = np.linspace(x2_min, x2_max, n_grid)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    # Create prediction grid
    n_pred = n_grid * n_grid
    grid_data = model.data.iloc[[0] * n_pred].copy()
    grid_data[var1] = X1.ravel()
    grid_data[var2] = X2.ravel()

    # Predict
    pred = model.predict(grid_data, scale="response")
    Z = pred.reshape((n_grid, n_grid))

    # Plot
    im = ax.contourf(X1, X2, Z, levels=20, cmap=cmap)
    ax.contour(X1, X2, Z, levels=10, colors="black", alpha=0.3, linewidths=0.5)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title(f"Smooth term: {var1} × {var2}")

    plt.colorbar(im, ax=ax, label="Effect")

    return ax


def plot_residuals(
    model: GAM,
    ax_array: np.ndarray | None = None,
) -> np.ndarray:
    """Plot residual diagnostics (4-panel).

    Args:
        model: Fitted GAM.
        ax_array: Array of 4 matplotlib axes.

    Returns:
        Array of axes.
    """
    if ax_array is None:
        fig, ax_array = plt.subplots(2, 2, figsize=(10, 8))
        ax_array = ax_array.ravel()

    # Compute residuals
    fitted = (
        model.fitted_values
        if hasattr(model, "fitted_values")
        else model.predict(model.data, scale="response")
    )
    residuals = model.data[model.formula.split("~")[0].strip()].values - fitted

    # Panel 1: Residuals vs Fitted
    ax_array[0].scatter(fitted, residuals, alpha=0.5, s=20)
    ax_array[0].axhline(y=0, color="r", linestyle="--")
    ax_array[0].set_xlabel("Fitted values")
    ax_array[0].set_ylabel("Residuals")
    ax_array[0].set_title("Residuals vs Fitted")
    ax_array[0].grid(True, alpha=0.3)

    # Panel 2: Q-Q plot
    from scipy import stats as sp_stats

    sp_stats.probplot(residuals, dist="norm", plot=ax_array[1])
    ax_array[1].set_title("Normal Q-Q Plot")
    ax_array[1].grid(True, alpha=0.3)

    # Panel 3: Scale-Location
    std_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
    ax_array[2].scatter(fitted, std_residuals, alpha=0.5, s=20)
    ax_array[2].set_xlabel("Fitted values")
    ax_array[2].set_ylabel("√|Standardized residuals|")
    ax_array[2].set_title("Scale-Location")
    ax_array[2].grid(True, alpha=0.3)

    # Panel 4: Histogram of residuals
    ax_array[3].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax_array[3].set_xlabel("Residuals")
    ax_array[3].set_ylabel("Frequency")
    ax_array[3].set_title("Histogram of Residuals")
    ax_array[3].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    return ax_array


def plot_diagnostics(
    model: GAM,
    **kwargs,
) -> None:
    """Plot model diagnostics.

    Args:
        model: Fitted GAM.
        **kwargs: matplotlib kwargs.
    """
    plot_residuals(model)


def plot_gam(
    model: GAM,
    which: str = "all",
    var_names: list[str] | None = None,
) -> None:
    """Plot all smooth terms of GAM.

    Args:
        model: Fitted GAM.
        which: 'smooth' (smooth terms only), 'residuals', or 'all'.
        var_names: Specific variables to plot (default: all).
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib not installed. Install with: pip install matplotlib")

    if not model.fitted:
        raise RuntimeError("Model not fitted")

    # Extract smooth variable names from formula
    import re

    smooth_pattern = r"s\(([^)]+)\)"
    matches = re.findall(smooth_pattern, model.formula)

    if var_names is None:
        var_names = matches

    if which in ["smooth", "all"]:
        n_smooths = len(var_names)
        n_cols = min(2, n_smooths)
        n_rows = (n_smooths + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_smooths == 1:
            axes = np.array([axes])
        else:
            axes = axes.ravel()

        for i, var_name in enumerate(var_names):
            if var_name in model.data.columns:
                plot_smooth(model, var_name, ax=axes[i])

        plt.tight_layout()
        plt.show()

    if which in ["residuals", "all"]:
        plot_residuals(model)
        plt.show()
