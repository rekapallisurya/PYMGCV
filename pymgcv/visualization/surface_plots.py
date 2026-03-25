"""3D surface plotting for tensor products."""

from __future__ import annotations

import numpy as np


def plot_surface(
    model,
    smooth_idx: int = 0,
    n_grid: int = 40,
    ax=None,
    **kwargs,
):
    """Plot a tensor product smooth as a 3D surface.

    Args:
        model: Fitted GAM instance with a te()/ti() smooth.
        smooth_idx: Index of the tensor product smooth term.
        n_grid: Grid resolution per axis.
        ax: Matplotlib 3D axes (creates one if None).
        **kwargs: Passed to ax.plot_surface().

    Returns:
        Matplotlib 3D axes.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    if not model.fitted:
        raise ValueError("Model must be fitted before plotting")

    mm = model.model_matrix
    if smooth_idx >= len(mm.smooth_indices):
        raise IndexError(f"smooth_idx={smooth_idx} >= number of smooths")

    smooth_spec = mm.smooth_specs_used[smooth_idx]
    if len(smooth_spec.variables) < 2:
        raise ValueError("3D surface requires te/ti smooth with ≥2 variables")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    var1, var2 = smooth_spec.variables[:2]
    x1_data = model.data[var1].values.astype(float)
    x2_data = model.data[var2].values.astype(float)

    x1_grid = np.linspace(x1_data.min(), x1_data.max(), n_grid)
    x2_grid = np.linspace(x2_data.min(), x2_data.max(), n_grid)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    import pandas as pd

    pred_df = pd.DataFrame({var1: X1.ravel(), var2: X2.ravel()})
    resp = getattr(mm, "formula_parser", None)
    response_col = model.formula.split("~")[0].strip()
    for col in model.data.columns:
        if col in (var1, var2, response_col):
            continue
        val = model.data[col].values
        if val.dtype.kind in ("U", "O", "S"):
            pred_df[col] = val[0]
        else:
            pred_df[col] = float(np.median(val))

    preds = model.predict(pred_df, scale="link")
    Z = preds.reshape(n_grid, n_grid)

    ax.plot_surface(X1, X2, Z, **kwargs)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_zlabel("Partial effect")
    ax.set_title(f"Tensor product: {smooth_spec.label}")

    return ax
