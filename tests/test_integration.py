"""Integration tests for PyMGCV GAM framework.

Demonstrates full workflow: data generation → model fitting → evaluation.
"""

import numpy as np
import pytest


class TestGAMIntegration:
    """Integration tests for complete GAM fitting workflow."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        np.random.seed(42)
        n = 200
        X = np.linspace(0, 2*np.pi, n)
        
        # True function: combination of sine and polynomial
        y_true = np.sin(X) + 0.1 * X**2
        
        # Add noise
        noise = np.random.normal(0, 0.1, n)
        y = y_true + noise
        
        return {'X': X, 'y': y, 'y_true': y_true}

    def test_bspline_basis_shape(self, synthetic_data):
        """Test B-spline basis has correct shape."""
        from pymgcv.smooth.bspline import BSplineBasis
        
        X = synthetic_data['X']
        basis = BSplineBasis(X, k=10, order=4)
        
        assert basis.basis_matrix.shape == (len(X), 10)
        assert np.all(np.isfinite(basis.basis_matrix))

    def test_penalty_matrix_properties(self, synthetic_data):
        """Test penalty matrix is symmetric."""
        from pymgcv.smooth.bspline import BSplineBasis
        
        X = synthetic_data['X']
        basis = BSplineBasis(X, k=10, order=4)
        S = basis.penalty_matrix
        
        # Symmetric
        assert np.allclose(S, S.T)

    def test_pspline_difference_penalty(self, synthetic_data):
        """Test P-spline difference penalties."""
        from pymgcv.smooth.pspline import difference_penalty
        
        # Test different orders
        for order in [1, 2, 3]:
            S = difference_penalty(k=10, order=order)
            assert S.shape == (10, 10)
            assert np.allclose(S, S.T)  # Symmetric
            
            # At least one eigenvalue should be 0 (constant function)
            eigenvalues = np.linalg.eigvalsh(S)
            assert np.min(eigenvalues) < 1e-10

    def test_aic_ubre_calculation(self, synthetic_data):
        """Test AIC and UBRE computation."""
        from pymgcv.criterions.aic_ubre import compute_aic, compute_ubre
        
        residuals = np.array([0.1, 0.2, -0.1, 0.05])
        edf = 5.0
        
        aic = compute_aic(residuals, edf, n=4)
        ubre = compute_ubre(residuals, edf, sigma2=0.01, n=4)
        
        assert np.isfinite(aic)
        assert np.isfinite(ubre)

    def test_family_gaussian(self, synthetic_data):
        """Test Gaussian family."""
        from pymgcv.distributions.family_base import GaussianFamily

        family = GaussianFamily()

        # Test link function (identity)
        eta = np.array([0.0, 1.0, -1.0])
        mu = family.linkinv(eta)
        assert np.allclose(mu, eta)  # Identity link

        # Test variance
        var = family.variance(mu)
        assert np.allclose(var, np.ones_like(mu))  # Constant variance

    def test_family_binomial(self, synthetic_data):
        """Test Binomial family."""
        from pymgcv.distributions.family_base import BinomialFamily

        family = BinomialFamily()

        # Test link function (logit → probabilities)
        eta = np.array([0.0, 1.0, -1.0])
        mu = family.linkinv(eta)
        assert np.all((mu >= 0) & (mu <= 1))  # Probabilities

        # Test variance
        var = family.variance(mu)
        assert np.all(var >= 0)

    def test_sparse_block_construction(self):
        """Test sparse block matrix assembly."""
        from pymgcv.linalg.sparse_utils import build_block_matrix
        
        # Create blocks
        A = np.eye(3)
        B = np.ones((3, 2))
        
        blocks = [[A, B]]
        H = build_block_matrix(blocks)
        
        assert H.shape == (3, 5)
        assert np.allclose(H[:, :3], A)
        assert np.allclose(H[:, 3:], B)

    def test_configuration_defaults(self):
        """Test package configuration."""
        from pymgcv import config
        
        assert hasattr(config, 'DEFAULT_K')
        assert hasattr(config, 'DEFAULT_ORDER')
        assert hasattr(config, 'TOLERANCE')
        assert config.DEFAULT_K >= 5
        assert config.DEFAULT_ORDER >= 3

    def test_gam_basic_fit(self, synthetic_data):
        """Test basic GAM fitting workflow."""
        from pymgcv.api.gam import GAM
        import pandas as pd

        X = synthetic_data['X']
        y = synthetic_data['y']
        data = pd.DataFrame({'x1': X, 'y': y})

        gam = GAM('y ~ s(x1)', data=data, family='gaussian')
        gam.fit()

        fitted = gam.predict(data, scale='response')
        residuals = y - fitted
        rmse = np.sqrt(np.mean(residuals ** 2))

        assert rmse < 0.3
        assert len(gam.beta) > 0

    def test_thin_plate_spline(self, synthetic_data):
        """Test thin-plate spline basis."""
        from pymgcv.smooth.thin_plate import ThinPlateSpline

        X = synthetic_data['X']
        tps = ThinPlateSpline(X.reshape(-1, 1), k=10)

        B = tps.basis_matrix()
        assert B.shape[0] == len(X)
        assert B.shape[1] <= 10
        assert np.all(np.isfinite(B))


class TestNumericalStability:
    """Test numerical stability and accuracy."""

    def test_no_nan_in_basis(self):
        """Ensure no NaN values in basis computation."""
        np.random.seed(42)
        X = np.linspace(-10, 10, 100)
        
        from pymgcv.smooth.bspline import BSplineBasis
        basis = BSplineBasis(X, k=15, order=4)
        
        assert not np.any(np.isnan(basis.basis_matrix))
        assert not np.any(np.isinf(basis.basis_matrix))

    def test_penalty_matrix_rank(self):
        """Test penalty matrix has expected rank."""
        from pymgcv.smooth.pspline import difference_penalty
        
        k = 10
        for order in [1, 2, 3]:
            S = difference_penalty(k, order=order)
            rank = np.linalg.matrix_rank(S)
            
            # Rank should be k - order
            expected_rank = k - order
            assert rank == expected_rank or rank == expected_rank - 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point(self):
        """Test with single data point."""
        X = np.array([0.5])
        
        from pymgcv.smooth.bspline import BSplineBasis
        try:
            basis = BSplineBasis(X, k=3, order=3)
            assert basis.basis_matrix.shape[0] == 1
        except ValueError:
            pass  # Expected for small sample

    def test_repeated_x_values(self):
        """Test with repeated X values."""
        X = np.array([0, 0, 1, 1, 2, 2])
        
        from pymgcv.smooth.bspline import BSplineBasis
        basis = BSplineBasis(X, k=5, order=3)
        
        # Should still produce valid basis
        assert basis.basis_matrix.shape[0] == len(X)
        assert not np.any(np.isnan(basis.basis_matrix))

    def test_large_k(self):
        """Test with many basis functions."""
        X = np.linspace(0, 1, 100)
        
        from pymgcv.smooth.bspline import BSplineBasis
        basis = BSplineBasis(X, k=50, order=4)
        
        assert basis.basis_matrix.shape == (100, 50)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
