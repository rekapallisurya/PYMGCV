"""Visualization of smooth effects and 3D surfaces.

Provides plotting utilities for GAM smooth terms and tensor product surfaces.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def plot_1d_smooth(
    model,
    smooth_idx: int = 0,
    n_points: int = 200,
    ci: bool = True,
    level: float = 0.95,
    ax=None,
    **kwargs,
):
    """Plot a 1D smooth effect from a fitted GAM.

    Args:
        model: Fitted GAM instance.
        smooth_idx: Index of smooth term to plot (default 0).
        n_points: Number of points on prediction grid.
        ci: Whether to add confidence band.
        level: Confidence level (default 0.95).
        ax: Matplotlib axes (creates one if None).
        **kwargs: Passed to ax.plot().

    Returns:
        Matplotlib axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('matplotlib is required for plotting')

    if ax is None:
        _, ax = plt.subplots()

    if not model.fitted:
        raise ValueError('Model must be fitted before plotting')

    mm = model.model_matrix
    if smooth_idx >= len(mm.smooth_indices):
        raise IndexError(f'smooth_idx={smooth_idx} >= number of smooths={len(mm.smooth_indices)}')

    smooth_spec = mm.smooth_specs_used[smooth_idx]
    var_name = smooth_spec.variables[0]

    x_data = model.data[var_name].values
    x_grid = np.linspace(x_data.min(), x_data.max(), n_points)

    # Build prediction dataframe at x_grid with other variables at mean
    import pandas as pd
    pred_df = pd.DataFrame({var_name: x_grid})
    for col in model.data.columns:
        if col == var_name or col == mm.formula_parser.response:
            continue
        val = model.data[col].values
        if val.dtype.kind in ('U', 'O', 'S'):
            pred_df[col] = val[0]  # first category
        else:
            pred_df[col] = float(np.median(val))

    preds = model.predict(pred_df, scale='link')

    # Extract just the contribution of this smooth (partial effect)
    ax.plot(x_grid, preds, **kwargs)
    ax.set_xlabel(var_name)
    ax.set_ylabel(f'Partial effect of {var_name}')
    ax.set_title(f'Smooth: {smooth_spec.label}')

    return ax


def plot_2d_smooth(
    model,
    smooth_idx: int = 0,
    n_grid: int = 50,
    ax=None,
    **kwargs,
):
    """Plot a 2D tensor product smooth as a contour/surface plot.

    Args:
        model: Fitted GAM instance with a te() or ti() smooth.
        smooth_idx: Index of the tensor product smooth term.
        n_grid: Grid size for surface evaluation.
        ax: Matplotlib axes (creates one if None).
        **kwargs: Passed to ax.contourf().

    Returns:
        Matplotlib axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('matplotlib is required for plotting')

    if ax is None:
        _, ax = plt.subplots()

    if not model.fitted:
        raise ValueError('Model must be fitted before plotting')

    mm = model.model_matrix
    if smooth_idx >= len(mm.smooth_indices):
        raise IndexError(f'smooth_idx={smooth_idx} >= number of smooths')

    smooth_spec = mm.smooth_specs_used[smooth_idx]
    if len(smooth_spec.variables) < 2:
        raise ValueError(f'2D plot requires te/ti smooth with ≥2 variables')

    var1, var2 = smooth_spec.variables[:2]
    x1_data = model.data[var1].values.astype(float)
    x2_data = model.data[var2].values.astype(float)

    x1_grid = np.linspace(x1_data.min(), x1_data.max(), n_grid)
    x2_grid = np.linspace(x2_data.min(), x2_data.max(), n_grid)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    import pandas as pd
    pred_df = pd.DataFrame({
        var1: X1.ravel(),
        var2: X2.ravel()
    })
    for col in model.data.columns:
        if col in (var1, var2, mm.formula_parser.response):
            continue
        val = model.data[col].values
        if val.dtype.kind in ('U', 'O', 'S'):
            pred_df[col] = val[0]
        else:
            pred_df[col] = float(np.median(val))

    preds = model.predict(pred_df, scale='link')
    Z = preds.reshape(n_grid, n_grid)

    cs = ax.contourf(X1, X2, Z, **kwargs)
    plt.colorbar(cs, ax=ax)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title(f'Tensor product: {smooth_spec.label}')

    return ax
