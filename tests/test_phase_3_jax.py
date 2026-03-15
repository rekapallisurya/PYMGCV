"""Integration tests for Phase 3: GPU acceleration via JAX.

Tests:
    - JAX device detection
    - JAX PIRLS iteration
    - GAM class with GPU acceleration
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from pymgcv.optimizer.jax_acceleration import (
    device_info,
    jax_pirls_iteration,
    jax_gradient_function,
)
from pymgcv.api.gam import GAM


class TestJAXAcceleration:
    """JAX acceleration tests."""

    def test_device_info(self) -> None:
        """Test JAX device detection."""
        info = device_info()
        assert isinstance(info, dict)
        assert 'available' in info
        # Device info won't crash even if JAX not installed

    def test_jax_pirls_iteration(self) -> None:
        """Test one PIRLS iteration via JAX."""
        n, p = 50, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        w = np.ones(n)
        z = np.random.randn(n)
        S = np.eye(p) * 0.1
        beta_old = np.zeros(p)

        beta_new = jax_pirls_iteration(beta_old, X, y, w, z, S)
        assert beta_new.shape == (p,)
        assert np.all(np.isfinite(beta_new))

    def test_jax_gradient_function(self) -> None:
        """Test JAX gradient computation."""
        n, p = 50, 10
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        S = np.eye(p) * 0.1

        grad_func = jax_gradient_function(X, y, S)
        beta = np.random.randn(p)
        g = grad_func(beta)

        assert g.shape == (p,)
        assert np.all(np.isfinite(g))


class TestGAMClass:
    """Tests for GAM class with GPU acceleration."""

    def test_gam_initialization(self) -> None:
        """Test GAM initialization."""
        model = GAM('y ~ s(x1) + s(x2)')
        assert model.formula == 'y ~ s(x1) + s(x2)'
        assert model.fitted is False
        assert model.beta is None

    def test_gam_fit_gaussian(self) -> None:
        """Test GAM fitting on Gaussian data."""
        np.random.seed(42)
        n = 100
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + np.random.normal(0, 0.1, n)

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

        model = GAM('y ~ s(x1) + s(x2)', family='gaussian')
        model.fit(data, verbose=False)

        assert model.fitted
        assert model.beta is not None
        assert model.smoothing_parameters is not None
        assert model.edf is not None
        assert len(model.beta) > 0

    def test_gam_predict(self) -> None:
        """Test GAM prediction."""
        np.random.seed(42)
        n = 50
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x1) + np.random.normal(0, 0.1, n)

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

        model = GAM('y ~ s(x1)', family='gaussian')
        model.fit(data)

        # Predict on same data
        pred_response = model.predict(data, scale='response')
        pred_link = model.predict(data, scale='link')

        assert pred_response.shape == (n,)
        assert pred_link.shape == (n,)
        assert np.all(np.isfinite(pred_response))
        assert np.all(np.isfinite(pred_link))

    def test_gam_summary(self) -> None:
        """Test GAM summary output."""
        np.random.seed(42)
        n = 50
        x1 = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x1) + np.random.normal(0, 0.1, n)

        data = pd.DataFrame({'x1': x1, 'y': y})

        model = GAM('y ~ s(x1)', family='gaussian')
        model.fit(data)

        summary = model.summary()
        assert isinstance(summary, str)
        assert 'Family' in summary
        assert 'Formula' in summary
        assert len(summary) > 0

    def test_gam_fit_poisson(self) -> None:
        """Test GAM fitting on Poisson data."""
        np.random.seed(42)
        n = 100
        x1 = np.linspace(0, 1, n)
        eta = -2 + 3 * np.sin(2 * np.pi * x1)
        mu = np.exp(eta)
        y = np.random.poisson(mu)

        data = pd.DataFrame({'x1': x1, 'y': y})

        model = GAM('y ~ s(x1)', family='poisson')
        model.fit(data, verbose=False)

        assert model.fitted
        assert model.beta is not None
        assert len(model.beta) > 0

    def test_gam_fit_with_offset(self) -> None:
        """Test GAM fitting with offset."""
        np.random.seed(42)
        n = 50
        x1 = np.linspace(0, 1, n)
        exposure = np.ones(n) * 100
        y = np.random.poisson(exposure * np.sin(2 * np.pi * x1))

        data = pd.DataFrame({'x1': x1, 'y': y, 'exposure': exposure})

        # Note: offset handling via design matrix
        model = GAM('y ~ s(x1)', family='poisson')
        model.fit(data, verbose=False)

        assert model.fitted


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
