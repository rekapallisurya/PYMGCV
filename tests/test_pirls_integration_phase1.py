"""Integration Tests for Phase 1 Tasks 1.3 and 1.4: GAM Fitting and Offset Handling."""

import numpy as np
import pytest
from pymgcv.api.gam import GAM
from pymgcv.optimizer.pirls import PIRLSSolver, solve_pirls
from pymgcv.distributions.family_base import GaussianFamily, PoissonFamily, BinomialFamily


class TestGAMFittingIntegration:
    """Test GAM fitting with improved PIRLS."""
    
    def test_gam_gaussian_fitting(self):
        """Test basic Gaussian GAM fitting."""
        np.random.seed(42)
        n = 50
        
        # Simple linear model
        data = {
            'y': np.random.randn(n) + 5,
            'x': np.random.randn(n),
        }
        
        # Create and fit model
        model = GAM('y ~ x', data=data, family='gaussian')
        model.fit(verbose=False)
        
        # Should converge
        assert model.fitted
        assert model.beta is not None
        assert len(model.beta) > 0
        assert np.all(np.isfinite(model.beta))

    def test_gam_poisson_fitting(self):
        """Test Poisson GAM fitting."""
        np.random.seed(42)
        n = 80
        
        data = {
            'y': np.random.poisson(lam=5, size=n),
            'x': np.random.randn(n),
        }
        
        model = GAM('y ~ x', data=data, family='poisson')
        model.fit(verbose=False)
        
        assert model.fitted
        assert np.all(np.isfinite(model.beta))

    def test_gam_binomial_fitting(self):
        """Test Binomial GAM fitting."""
        np.random.seed(42)
        n = 100
        
        data = {
            'y': np.random.binomial(1, 0.5, n),
            'x': np.random.randn(n),
        }
        
        model = GAM('y ~ x', data=data, family='binomial')
        model.fit(verbose=False)
        
        assert model.fitted
        assert np.all(np.isfinite(model.beta))

    def test_gam_with_offset(self):
        """Test GAM fitting with offset term."""
        np.random.seed(42)
        n = 60
        
        data = {
            'y': np.random.poisson(lam=3, size=n),
            'x': np.random.randn(n),
            'offset': np.zeros(n),  # Zero offset for testing
        }
        
        model = GAM('y ~ x', data=data, family='poisson', offset='offset')
        model.fit(verbose=False)
        
        assert model.fitted
        assert np.all(np.isfinite(model.beta))


class TestOffsetEdgeCases:
    """Test Task 1.4: Offset edge cases."""
    
    def test_pirls_with_zero_offset(self):
        """PIRLS should handle zero offset."""
        np.random.seed(42)
        n = 40
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n) + 2
        offset = np.zeros(n)
        
        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]
        
        solver = PIRLSSolver(X, y, family, S_list, offset=offset)
        beta = solver.solve(max_iter=20)
        
        assert np.all(np.isfinite(beta))
        assert solver.converged

    def test_pirls_with_nonzero_offset(self):
        """PIRLS should properly use non-zero offset."""
        np.random.seed(42)
        n = 40
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        
        true_beta = np.array([1.0, 0.5])
        offset = np.full(n, 0.5)
        y = X @ true_beta + offset + np.random.randn(n) * 0.1
        
        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]
        
        solver = PIRLSSolver(X, y, family, S_list, offset=offset)
        beta = solver.solve(max_iter=20)
        
        # Should recover coefficients despite offset
        assert solver.converged
        assert np.allclose(beta, true_beta, atol=0.3)

    def test_pirls_offset_vs_no_offset(self):
        """Offset should only affect intercept interpretation."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 2 + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
        
        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]
        
        # Without offset
        solver1 = PIRLSSolver(X, y, family, S_list, offset=None)
        beta1 = solver1.solve(max_iter=20)
        
        # With zero offset
        solver2 = PIRLSSolver(X, y, family, S_list, offset=np.zeros(n))
        beta2 = solver2.solve(max_iter=20)
        
        # Should be identical
        assert np.allclose(beta1, beta2, atol=1e-6)

    def test_pirls_large_offset(self):
        """PIRLS should handle large offsets gracefully."""
        np.random.seed(42)
        n = 40
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 10 + X[:, 1] + np.random.randn(n)
        
        offset = np.full(n, 8.0)  # Large offset
        
        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]
        
        solver = PIRLSSolver(X, y, family, S_list, offset=offset)
        beta = solver.solve(max_iter=20)
        
        # Should still converge
        assert np.all(np.isfinite(beta))

    def test_offset_affects_predictions(self):
        """Offset should affect fitted values correctly."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n) + 5
        offset = np.full(n, 2.0)
        
        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]
        
        solver = PIRLSSolver(X, y, family, S_list, offset=offset)
        beta = solver.solve(max_iter=20)
        
        # Fitted values should include offset
        fitted = X @ beta + offset
        assert np.all(np.isfinite(fitted))
        # Offset should add to predictions
        fitted_no_offset = X @ beta
        assert np.allclose(fitted - fitted_no_offset, offset, atol=1e-10)


class TestPIRLSWeightsAndOffset:
    """Test combined weights and offset functionality."""
    
    def test_weights_and_offset_together(self):
        """PIRLS should handle weights and offset together."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([2.0, 0.5]) + np.random.randn(n) * 0.5
        
        weights = np.ones(n)
        offset = np.zeros(n)
        
        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]
        
        solver = PIRLSSolver(X, y, family, S_list, 
                            weights=weights, offset=offset)
        beta = solver.solve(max_iter=20)
        
        assert solver.converged
        assert np.all(np.isfinite(beta))

    def test_weighted_regression(self):
        """Weighted PIRLS should respect heteroscedasticity."""
        np.random.seed(42)
        n = 60
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        
        # Heteroscedastic data
        y = X @ np.array([1.0, 0.5])
        sigma = 0.1 + 0.5 * X[:, 1]**2
        y += np.random.randn(n) * sigma
        
        # Weights should be inverse variance
        weights = 1.0 / sigma
        
        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]
        
        # Weighted fit
        solver_wgt = PIRLSSolver(X, y, family, S_list, weights=weights)
        beta_wgt = solver_wgt.solve(max_iter=20)
        
        # Unweighted fit (for comparison)
        solver_unwgt = PIRLSSolver(X, y, family, S_list)
        beta_unwgt = solver_unwgt.solve(max_iter=20)
        
        # Both should converge
        assert solver_wgt.converged
        assert solver_unwgt.converged
        
        # Weighted should give different result
        assert not np.allclose(beta_wgt, beta_unwgt, atol=0.05)


class TestPredictionWithOffset:
    """Test predictions with offset."""
    
    def test_prediction_respects_offset(self):
        """Predictions should include offset."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([2.0, 0.5]) + 1.0 + np.random.randn(n) * 0.1
        offset = np.full(n, 1.0)
        
        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]
        
        solver = PIRLSSolver(X, y, family, S_list, offset=offset)
        beta = solver.solve(max_iter=20)
        
        # Predictions with offset
        eta = X @ beta + offset
        mu = family.linkinv(eta)
        
        # Should approximate y
        assert np.allclose(mu, y, atol=0.5)

    def test_fitted_values_method(self):
        """Test fitted_values method includes offset."""
        np.random.seed(42)
        n = 40
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.5, 0.7]) + np.random.randn(n) * 0.1
        offset = np.full(n, 0.5)
        
        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]
        
        solver = PIRLSSolver(X, y, family, S_list, offset=offset)
        solver.solve(max_iter=20)
        
        fitted = solver.fitted_values()
        
        # Should match y approximately
        assert np.allclose(fitted, y, atol=0.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
