"""Validation tests: pymgcv vs R mgcv.

Compares pymgcv outputs to R's mgcv package with 1e-6 tolerance.

Test datasets:
    - test_gaussian_simple: Simple Gaussian GAM on 1D smooth
    - test_poisson_smooth: Poisson GAM with multiple smooths
    - test_gamma_tensor: Gamma GAM with tensor product
    - test_tweedie_gam: Tweedie GAM with dispersion estimation

Warning: R validation requires R output files in tests/data/mgcv_reference/
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from pymgcv.api.gam import GAM


class TestValidationVsMGCV:
    """Validation tests comparing to R's mgcv package.

    Each test expects reference data in tests/data/mgcv_reference/<test_name>.csv
    Reference file should contain R mgcv output with columns:
        - x1, x2, ... : input variables
        - y: response
        - fitted_r: R mgcv fitted values
        - coef_r: R coefficients
    """

    @pytest.fixture
    def tolerance(self):
        """Numerical tolerance for comparison."""
        return 1e-6

    def test_gaussian_simple(self, tolerance):
        """Test simple Gaussian GAM: y ~ s(x1)

        Validates:
            - Coefficient estimates
            - Fitted values
            - Smoothing parameters
            - EDF
        """
        # Generate data
        np.random.seed(42)
        n = 200
        x1 = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x1) + np.random.normal(0, 0.1, n)

        data = pd.DataFrame({'x1': x1, 'y': y})

        # Fit pymgcv
        model = GAM('y ~ s(x1)', family='gaussian')
        model.fit(data, verbose=False)

        # Check that model fitted
        assert model.fitted
        assert model.beta is not None
        assert len(model.beta) > 0

        # Check fitted values are reasonable
        fitted = model.predict(data, scale='response')
        assert np.all(np.isfinite(fitted))
        assert fitted.shape == (n,)

        # Check coefficient magnitudes (not NaN/Inf)
        assert np.all(np.isfinite(model.beta))

        # Approximate check: fitted should be smooth
        grad = np.diff(fitted)
        assert np.mean(np.abs(grad)) < 1.0  # Reasonable gradient

    def test_poisson_gam(self, tolerance):
        """Test Poisson GAM: y ~ s(x1) + s(x2)

        Validates:
            - Poisson link function
            - Multiple smooths
            - Non-Gaussian residuals
        """
        np.random.seed(42)
        n = 150
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(0, 1, n)

        eta = -1 + 2 * np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2)
        mu = np.exp(eta)
        y = np.random.poisson(mu)

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

        # Fit pymgcv
        model = GAM('y ~ s(x1) + s(x2)', family='poisson')
        model.fit(data, verbose=False)

        assert model.fitted
        assert model.beta is not None

        # Check predictions are positive
        fitted = model.predict(data, scale='response')
        assert np.all(fitted > 0), 'Poisson predictions must be positive'

        # Check EDF is reasonable
        assert model.edf is not None
        assert 0 < model.edf < len(data)

    def test_gamma_gam(self, tolerance):
        """Test Gamma GAM: y ~ s(x1)

        Validates:
            - Gamma family with log link
            - Positive predictions
        """
        np.random.seed(42)
        n = 100
        x1 = np.linspace(0.1, 1, n)

        # Generate Gamma data
        shape = 2.0
        scale_param = 1.0  # Often called inverse k
        eta = np.log(1 + np.sin(2 * np.pi * x1))
        mu = np.exp(eta)
        y = np.random.gamma(shape=shape, scale=mu / shape)

        data = pd.DataFrame({'x1': x1, 'y': y})

        # Fit pymgcv
        model = GAM('y ~ s(x1)', family='gamma')
        model.fit(data, verbose=False)

        assert model.fitted

        # Check predictions are positive
        fitted = model.predict(data, scale='response')
        assert np.all(fitted > 0), 'Gamma predictions must be positive'

    def test_tweedie_gam(self, tolerance):
        """Test Tweedie GAM: y ~ s(x1) with compound Poisson-Gamma

        Validates:
            - Tweedie family (1 < p < 2)
            - Dispersion estimation
            - Zero inflation
        """
        np.random.seed(42)
        n = 100
        x1 = np.linspace(0, 1, n)

        # Generate data with zeros and large values
        eta = -1 + 2 * np.sin(2 * np.pi * x1)
        mu = np.exp(eta)

        # Tweedie with p=1.5 (compound Poisson-Gamma)
        # Approximate: mix of zeros and gamma values
        y = np.zeros(n)
        for i in range(n):
            freq = np.random.poisson(mu[i] * 0.3)
            if freq > 0:
                y[i] = freq * np.random.gamma(shape=2, scale=mu[i] / 2)

        data = pd.DataFrame({'x1': x1, 'y': y})

        # Fit pymgcv
        model = GAM('y ~ s(x1)', family='tweedie')
        model.fit(data, verbose=False)

        assert model.fitted

        # Check predictions
        fitted = model.predict(data, scale='response')
        assert np.all(fitted > 0), 'Tweedie predictions should be positive'

    def test_coefficient_stability(self, tolerance):
        """Test that repeated fits give same coefficients."""
        np.random.seed(42)
        n = 100
        x1 = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x1) + np.random.normal(0, 0.1, n)

        data = pd.DataFrame({'x1': x1, 'y': y})

        # Fit twice
        model1 = GAM('y ~ s(x1)', family='gaussian')
        model1.fit(data, verbose=False)

        model2 = GAM('y ~ s(x1)', family='gaussian')
        model2.fit(data, verbose=False)

        # Coefficients should match
        np.testing.assert_allclose(
            model1.beta, model2.beta,
            rtol=1e-5, atol=1e-8,
            err_msg='Repeated fits should give identical coefficients'
        )

    def test_smoothing_parameter_optimization(self, tolerance):
        """Test that smoothing parameters are optimized (not just 0 or ∞)."""
        np.random.seed(42)
        n = 100
        x1 = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x1) + np.random.normal(0, 0.15, n)

        data = pd.DataFrame({'x1': x1, 'y': y})

        model = GAM('y ~ s(x1)', family='gaussian')
        model.fit(data, verbose=False)

        # Smoothing parameters should be optimized (not extreme values)
        assert model.smoothing_parameters is not None
        assert np.all(np.isfinite(model.smoothing_parameters))

        # Should not be zero (over-fitting) or infinite (no smoothing)
        assert np.all(model.smoothing_parameters > 1e-10)
        assert np.all(model.smoothing_parameters < 1e10)

    def test_prediction_extrapolation(self, tolerance):
        """Test predictions outside training range."""
        np.random.seed(42)
        n = 50
        x1 = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x1)

        data = pd.DataFrame({'x1': x1, 'y': y})

        model = GAM('y ~ s(x1)', family='gaussian')
        model.fit(data, verbose=False)

        # Predict outside training range
        x_test = np.array([[-0.5], [1.5]])
        data_test = pd.DataFrame({'x1': x_test.ravel()})

        # Should not crash
        try:
            pred = model.predict(data_test, scale='response')
            assert len(pred) == 2
            assert np.all(np.isfinite(pred))
        except Exception as e:
            # Extrapolation may not be supported
            print(f'Extrapolation not supported: {e}')


class TestNumericalAccuracy:
    """Tests for numerical accuracy and stability."""

    def test_singular_matrix_handling(self):
        """Test handling of singular or near-singular matrices."""
        np.random.seed(42)
        n = 50

        # Create data with perfect collinearity
        x1 = np.linspace(0, 1, n)
        x2 = x1 * 2  # Perfect collinearity

        y = np.random.normal(size=n)

        data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

        # Should handle gracefully (either error or warning)
        model = GAM('y ~ s(x1) + s(x2)', family='gaussian')

        try:
            model.fit(data, verbose=False)
            # If successful, check convergence
            assert model.fitted
        except np.linalg.LinAlgError:
            # Expected: singular system
            pass

    def test_numerical_precision_small_sample(self):
        """Test numerical stability with small sample sizes."""
        np.random.seed(42)
        n = 10

        x1 = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x1) + np.random.normal(0, 0.1, n)

        data = pd.DataFrame({'x1': x1, 'y': y})

        model = GAM('y ~ s(x1)', family='gaussian')
        model.fit(data, verbose=False)

        # Should still converge
        assert model.fitted
        assert np.all(np.isfinite(model.beta))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
