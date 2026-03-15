"""Tests for visualization modules: smooth_plots, surface_plots, diagnostics_plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymgcv.api.gam import GAM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_1d_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)
    return pd.DataFrame({'x': x, 'y': y})


def _gaussian_2d_data(n=150, seed=7):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + rng.normal(0, 0.2, n)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})


@pytest.fixture(scope='module')
def fitted_1d():
    df = _gaussian_1d_data()
    m = GAM('y ~ s(x)', data=df, family='gaussian')
    m.fit()
    return m, df


@pytest.fixture(scope='module')
def fitted_2d():
    df = _gaussian_2d_data()
    m = GAM('y ~ te(x1, x2)', data=df, family='gaussian')
    m.fit()
    return m, df


# ---------------------------------------------------------------------------
# smooth_plots tests
# ---------------------------------------------------------------------------

class TestPlot1dSmooth:

    def test_returns_axes(self, fitted_1d):
        pytest.importorskip('matplotlib')
        from pymgcv.visualization.smooth_plots import plot_1d_smooth
        model, _ = fitted_1d
        ax = plot_1d_smooth(model, smooth_idx=0)
        import matplotlib.axes as mpl_axes
        assert isinstance(ax, mpl_axes.Axes)

    def test_unfitted_raises(self):
        pytest.importorskip('matplotlib')
        from pymgcv.visualization.smooth_plots import plot_1d_smooth
        m = GAM('y ~ s(x)', data=_gaussian_1d_data(), family='gaussian')
        with pytest.raises(ValueError, match='fitted'):
            plot_1d_smooth(m)

    def test_bad_index_raises(self, fitted_1d):
        pytest.importorskip('matplotlib')
        from pymgcv.visualization.smooth_plots import plot_1d_smooth
        model, _ = fitted_1d
        with pytest.raises(IndexError):
            plot_1d_smooth(model, smooth_idx=999)

    def test_n_points_controls_resolution(self, fitted_1d):
        pytest.importorskip('matplotlib')
        from pymgcv.visualization.smooth_plots import plot_1d_smooth
        import matplotlib.pyplot as plt
        model, _ = fitted_1d
        ax = plot_1d_smooth(model, smooth_idx=0, n_points=50)
        line = ax.lines[0]
        assert len(line.get_xdata()) == 50
        plt.close('all')

    def test_accepts_existing_axes(self, fitted_1d):
        """Should use provided axes instead of creating new figure."""
        pytest.importorskip('matplotlib')
        import matplotlib.pyplot as plt
        from pymgcv.visualization.smooth_plots import plot_1d_smooth
        model, _ = fitted_1d
        _, existing_ax = plt.subplots()
        returned_ax = plot_1d_smooth(model, smooth_idx=0, ax=existing_ax)
        assert returned_ax is existing_ax
        plt.close('all')


# ---------------------------------------------------------------------------
# surface_plots tests
# ---------------------------------------------------------------------------

class TestPlotSurface:

    def test_returns_3d_axes(self, fitted_2d):
        pytest.importorskip('matplotlib')
        from pymgcv.visualization.surface_plots import plot_surface
        model, _ = fitted_2d
        ax = plot_surface(model, smooth_idx=0, n_grid=10)
        # mpl_toolkits installs the Axes3D type
        assert 'Axes3D' in type(ax).__name__ or hasattr(ax, 'get_zlim')
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_unfitted_raises(self):
        pytest.importorskip('matplotlib')
        from pymgcv.visualization.surface_plots import plot_surface
        df = _gaussian_2d_data()
        m = GAM('y ~ te(x1, x2)', data=df, family='gaussian')
        with pytest.raises(ValueError, match='fitted'):
            plot_surface(m)

    def test_1d_smooth_raises(self, fitted_1d):
        pytest.importorskip('matplotlib')
        from pymgcv.visualization.surface_plots import plot_surface
        model, _ = fitted_1d
        with pytest.raises(ValueError, match='2'):
            plot_surface(model, smooth_idx=0, n_grid=5)

    def test_bad_index_raises(self, fitted_2d):
        pytest.importorskip('matplotlib')
        from pymgcv.visualization.surface_plots import plot_surface
        model, _ = fitted_2d
        with pytest.raises(IndexError):
            plot_surface(model, smooth_idx=999)


# ---------------------------------------------------------------------------
# diagnostics_plots tests
# ---------------------------------------------------------------------------

class TestPlotModelDiagnostics:

    def test_returns_axes_array(self, fitted_1d):
        pytest.importorskip('matplotlib')
        from pymgcv.visualization.diagnostics_plots import plot_model_diagnostics
        model, _ = fitted_1d
        axes = plot_model_diagnostics(model)
        assert axes.shape == (2, 2)
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_unfitted_raises(self):
        pytest.importorskip('matplotlib')
        from pymgcv.visualization.diagnostics_plots import plot_model_diagnostics
        m = GAM('y ~ s(x)', data=_gaussian_1d_data(), family='gaussian')
        with pytest.raises(ValueError, match='fitted'):
            plot_model_diagnostics(m)

    def test_accepts_existing_axes(self, fitted_1d):
        pytest.importorskip('matplotlib')
        import matplotlib.pyplot as plt
        from pymgcv.visualization.diagnostics_plots import plot_model_diagnostics
        model, _ = fitted_1d
        _, axes = plt.subplots(2, 2)
        returned = plot_model_diagnostics(model, axes=axes)
        assert returned is axes
        plt.close('all')

    def test_four_plots_populated(self, fitted_1d):
        """Each of the 4 axes should have been drawn on."""
        pytest.importorskip('matplotlib')
        import matplotlib.pyplot as plt
        from pymgcv.visualization.diagnostics_plots import plot_model_diagnostics
        model, _ = fitted_1d
        axes = plot_model_diagnostics(model)
        for ax in axes.ravel():
            # at least one artist should exist on each subplot
            assert ax.has_data(), f'axes {ax.get_title()!r} has no data'
        plt.close('all')
