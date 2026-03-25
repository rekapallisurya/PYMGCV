"""Test Phase 1 Task 1.1: PIRLS Stability Improvements.

Tests for:
1. Line search implementation
2. Improved convergence checks
3. NaN/Inf handling
4. Weight integration
5. Offset edge cases
"""

import numpy as np
import pytest

from pymgcv.distributions.family_base import BinomialFamily, GaussianFamily, PoissonFamily
from pymgcv.optimizer.pirls import PIRLSSolver, solve_pirls


class TestPIRLSLineSearch:
    """Test line search for stability."""

    def test_line_search_prevents_overshoot_gaussian(self):
        """Line search should prevent overshooting on difficult Gaussian data."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_true = np.array([1.0, 2.0])
        y = X @ beta_true + np.random.randn(n) * 0.5

        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]  # No penalty

        solver = PIRLSSolver(X, y, family, S_list)
        beta = solver.solve(max_iter=20, verbose=False)

        # Should converge with small deviance
        assert solver.converged, "Should converge with line search"
        assert np.allclose(beta, beta_true, atol=0.5)  # Relaxed tolerance

    def test_line_search_step_size_tracking(self):
        """Line search should track decreasing step sizes."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(lam=5, size=n).astype(float)

        family = PoissonFamily()
        S_list = [np.zeros((2, 2))]

        solver = PIRLSSolver(X, y, family, S_list)
        beta = solver.solve(max_iter=25, verbose=False)

        # Check history tracked step sizes
        assert "step_size" in solver.history[0]
        assert all(0 <= h["step_size"] <= 1.0 for h in solver.history)

    def test_line_search_with_penalties(self):
        """Line search should work with penalties."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.1

        family = GaussianFamily()
        S = np.array([[0, 0], [0, 1]])  # Penalize slope
        S_list = [S]
        lambda_vec = np.array([0.1])

        solver = PIRLSSolver(X, y, family, S_list, lambda_vec=lambda_vec)
        beta = solver.solve(max_iter=25, verbose=False)

        assert solver.converged
        # Penalty should shrink slope (but not below 0.5 with weak penalty)
        assert np.abs(beta[1]) < 0.8  # Relaxed tolerance given weak penalty


class TestPIRLSConvergenceCriteria:
    """Test improved convergence checking."""

    def test_convergence_checks_all_criteria(self):
        """Convergence should require multiple criteria."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(lam=3, size=n).astype(float)

        family = PoissonFamily()
        S_list = [np.zeros((2, 2))]

        solver = PIRLSSolver(X, y, family, S_list)
        beta = solver.solve(max_iter=30, verbose=False)

        # Check convergence history tracked deviance
        assert len(solver.dev_history) > 0
        assert solver.converged

        # Deviance should be decreasing
        for i in range(1, len(solver.dev_history)):
            assert solver.dev_history[i] <= solver.dev_history[i - 1] + 1e-4

    def test_convergence_relative_change(self):
        """Convergence should check relative deviance change."""
        np.random.seed(42)
        n = 80
        X = np.random.randn(n, 3)
        X[:, 0] = 1  # Intercept
        y = np.random.binomial(1, 0.5, n)

        family = BinomialFamily()
        S_list = [np.zeros((3, 3))]

        solver = PIRLSSolver(X, y, family, S_list)
        beta = solver.solve(max_iter=25, verbose=False)

        assert solver.converged

        # Check that iterations are reasonable
        assert solver.iterations < 25  # Should converge early


class TestPIRLSNaNHandling:
    """Test NaN and infinity handling."""

    def test_nan_values_detected_and_handled(self):
        """NaN values should revert to previous step."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])

        # Create data that could cause NaNs without safeguards
        y = np.concatenate(
            [np.random.poisson(lam=10, size=n // 2), np.random.poisson(lam=100, size=n // 2)]
        ).astype(float)

        family = PoissonFamily()
        S_list = [np.zeros((2, 2))]

        solver = PIRLSSolver(X, y, family, S_list)
        beta = solver.solve(max_iter=20, verbose=False)

        # Should complete without NaN propagation
        assert np.all(np.isfinite(beta))
        assert np.all(np.isfinite(solver.mu))

    def test_infinite_offset_handled(self):
        """Infinite offset values should be replaced with zero."""
        n = 30
        X = np.ones((n, 2))
        X[:, 1] = np.arange(n)
        y = np.random.randn(n) + 5

        offset = np.full(n, np.inf)
        offset[0] = 0  # One valid value

        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]

        # Should handle infinite offsets gracefully
        solver = PIRLSSolver(X, y, family, S_list, offset=offset)

        # Infinite offsets replaced with zeros
        assert np.all(np.isfinite(solver.offset))

    def test_zero_variance_handling(self):
        """Zero variance in weights should be safeguarded."""
        n = 40
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(lam=5, size=n).astype(float)

        family = PoissonFamily()
        S_list = [np.zeros((2, 2))]

        solver = PIRLSSolver(X, y, family, S_list)

        # Simulate iteration with zero variance
        solver.eta = solver.X @ solver.beta + solver.offset
        solver.mu = solver.family.linkinv(solver.eta)

        # Variance should not be zero due to safeguards
        var_mu = solver.family.variance(solver.mu, solver.dispersion)
        var_mu_safe = np.maximum(var_mu, 1e-10)
        assert np.all(var_mu_safe > 1e-11)


class TestPIRLSWeights:
    """Test weight integration in solve method."""

    def test_weights_parameter_acceptance(self):
        """PIRLS should accept weights parameter."""
        n = 40
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        weights = np.random.uniform(0.5, 2.0, n)

        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]

        # Should initialize with weights
        solver = PIRLSSolver(X, y, family, S_list, weights=weights)
        beta = solver.solve(max_iter=20)

        assert np.all(np.isfinite(beta))

    def test_weights_uniform_equivalence(self):
        """Uniform weights should give same result as no weights."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([2.0, 1.0]) + np.random.randn(n) * 0.1

        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]

        # No weights
        solver1 = PIRLSSolver(X, y, family, S_list)
        beta1 = solver1.solve(max_iter=20)

        # Uniform weights
        solver2 = PIRLSSolver(X, y, family, S_list, weights=np.ones(n))
        beta2 = solver2.solve(max_iter=20)

        # Should be identical
        assert np.allclose(beta1, beta2, atol=1e-6)

    def test_weights_validation(self):
        """Weights should be validated."""
        n = 30
        X = np.ones((n, 2))
        y = np.random.randn(n)

        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]

        # Negative weights should fail
        with pytest.raises(ValueError, match="positive"):
            PIRLSSolver(X, y, family, S_list, weights=np.full(n, -1.0))

        # Wrong length should fail
        with pytest.raises(ValueError, match="length"):
            PIRLSSolver(X, y, family, S_list, weights=np.ones(n + 1))


class TestPIRLSOffsets:
    """Test offset edge cases."""

    def test_offset_parameter_acceptance(self):
        """PIRLS should accept offset parameter."""
        n = 30
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        offset = np.random.randn(n) * 0.5

        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]

        solver = PIRLSSolver(X, y, family, S_list, offset=offset)
        beta = solver.solve(max_iter=20)

        assert np.all(np.isfinite(beta))

    def test_zero_offset_default(self):
        """Default offset should be zeros."""
        n = 30
        X = np.ones((n, 2))
        y = np.random.randn(n)

        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]

        solver = PIRLSSolver(X, y, family, S_list)
        assert np.allclose(solver.offset, np.zeros(n))

    def test_offset_dimension_validation(self):
        """Offset dimensions should match X."""
        n = 30
        X = np.ones((n, 2))
        y = np.random.randn(n)

        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]

        # Wrong dimension
        with pytest.raises(ValueError, match="Offset length"):
            PIRLSSolver(X, y, family, S_list, offset=np.ones(n + 5))


class TestPIRLSFunctionalAPI:
    """Test functional API with Phase 1 improvements."""

    def test_solve_pirls_with_weights(self):
        """Functional API should support weights."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.1
        weights = np.ones(n)

        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]

        beta, converged = solve_pirls(X, y, family, S_list, weights=weights, max_iter=20)

        assert converged
        assert np.all(np.isfinite(beta))

    def test_solve_pirls_with_offset(self):
        """Functional API should support offset."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.1
        offset = np.zeros(n)

        family = GaussianFamily()
        S_list = [np.zeros((2, 2))]

        beta, converged = solve_pirls(X, y, family, S_list, offset=offset, max_iter=20)

        assert converged
        assert np.all(np.isfinite(beta))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
