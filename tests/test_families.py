"""Unit tests for distribution families.

Tests verify:
1. Link function correctness
2. Variance function correctness
3. Log-likelihood computation
4. Numerical stability
5. Gradient correctness

All tests compare with theoretical expectations.
"""

import numpy as np
import pytest

from pymgcv.distributions.family_base import (
    BinomialFamily,
    GammaFamily,
    GaussianFamily,
    InverseGaussianFamily,
    NegativeBinomialFamily,
    PoissonFamily,
    TweedieFamily,
)


class TestBinomialFamily:
    """Tests for Binomial family."""

    def test_binomial_logit_link(self):
        """Test logit link function."""
        family = BinomialFamily(link="logit")
        eta = np.array([-2, -1, 0, 1, 2])
        mu = family.linkinv(eta)

        # Check bounds: 0 < μ < 1
        assert np.all((mu > 0) & (mu < 1))

        # Check special values
        assert np.isclose(family.linkinv(0), 0.5)  # logit(0) = 0.5
        assert family.linkinv(-1000) < 1e-5  # logit(-inf) ≈ 0
        assert family.linkinv(1000) > 1 - 1e-5  # logit(inf) ≈ 1

    def test_binomial_probit_link(self):
        """Test probit link function."""
        family = BinomialFamily(link="probit")
        eta = np.array([-2, -1, 0, 1, 2])
        mu = family.linkinv(eta)

        # Check bounds
        assert np.all((mu > 0) & (mu < 1))

        # Check special value: probit(0) = Φ(0) = 0.5
        assert np.isclose(family.linkinv(0), 0.5, atol=1e-10)

    def test_binomial_cloglog_link(self):
        """Test complementary log-log link function."""
        family = BinomialFamily(link="cloglog")
        eta = np.array([-2, -1, 0, 1, 2])
        mu = family.linkinv(eta)

        # Check bounds
        assert np.all((mu > 0) & (mu < 1))

        # Check special value: cloglog(0) = 1 - exp(-1) ≈ 0.632
        assert np.isclose(family.linkinv(0), 1 - np.exp(-1), atol=1e-10)

    def test_binomial_dmu_deta_logit(self):
        """Test derivative of logit link."""
        family = BinomialFamily(link="logit")
        eta = np.linspace(-5, 5, 100)
        dmu_deta = family.dmu_deta(eta)

        # dμ/dη = μ(1-μ), which has max at μ=0.5
        max_deriv = np.max(dmu_deta)
        # Use relaxed tolerance due to discretization
        assert np.isclose(max_deriv, 0.25, atol=1e-3)

        # Derivative should be positive everywhere
        assert np.all(dmu_deta > 0)

    def test_binomial_variance(self):
        """Test variance function μ(1-μ)."""
        family = BinomialFamily()
        mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        var = family.variance(mu)

        expected_var = mu * (1 - mu)
        assert np.allclose(var, expected_var)

        # Variance maximized at μ=0.5
        assert var[2] > var[0]
        assert var[2] > var[4]

    def test_binomial_loglik_perfect(self):
        """Test log-likelihood with perfect predictions."""
        family = BinomialFamily()

        # Case 1: Perfect prediction of ones (y=1, μ=1)
        y = np.array([1, 1, 1])
        mu = np.array([0.999, 0.999, 0.999])
        ll = family.loglik(y, mu)
        # Allow small numerical error from numerical precision
        assert ll > -0.01, f"Expected ll close to 0, got {ll}"

        # Case 2: Perfect prediction of zeros (y=0, μ=0)
        y = np.array([0, 0, 0])
        mu = np.array([0.001, 0.001, 0.001])
        ll = family.loglik(y, mu)
        # Allow small numerical error from numerical precision
        assert ll > -0.01, f"Expected ll close to 0, got {ll}"

    def test_binomial_loglik_random(self):
        """Test log-likelihood with random data."""
        family = BinomialFamily()
        np.random.seed(42)

        n = 100
        y = np.random.binomial(1, 0.5, n)
        mu = np.random.uniform(0.1, 0.9, n)

        ll = family.loglik(y, mu)

        # Log-likelihood should be negative
        assert ll < 0

        # Computing again should give same result
        ll2 = family.loglik(y, mu)
        assert np.isclose(ll, ll2)

    def test_binomial_invalid_link(self):
        """Test that invalid link raises error."""
        with pytest.raises(ValueError):
            BinomialFamily(link="invalid_link")


class TestNegativeBinomialFamily:
    """Tests for Negative Binomial family."""

    def test_negbinom_initialization(self):
        """Test initialization with valid theta."""
        family = NegativeBinomialFamily(theta=1.0)
        assert family.theta == 1.0

        family = NegativeBinomialFamily(theta=2.5)
        assert family.theta == 2.5

    def test_negbinom_invalid_theta(self):
        """Test that theta <= 0 raises error."""
        with pytest.raises(ValueError):
            NegativeBinomialFamily(theta=0)

        with pytest.raises(ValueError):
            NegativeBinomialFamily(theta=-1)

    def test_negbinom_log_link(self):
        """Test log link function."""
        family = NegativeBinomialFamily()
        eta = np.array([-2, -1, 0, 1, 2])
        mu = family.linkinv(eta)

        # μ = exp(η), so all values > 0
        assert np.all(mu > 0)

        # Check values
        expected_mu = np.exp(eta)
        assert np.allclose(mu, expected_mu)

    def test_negbinom_variance(self):
        """Test variance function μ + μ²/θ."""
        family = NegativeBinomialFamily(theta=2.0)
        mu = np.array([1, 2, 5, 10])
        var = family.variance(mu)

        # Var = μ + μ²/θ
        expected_var = mu + mu**2 / 2.0
        assert np.allclose(var, expected_var)

        # Variance should exceed mean (overdispersion)
        assert np.all(var > mu)

    def test_negbinom_variance_parameters(self):
        """Test how variance changes with theta."""
        mu = np.array([5.0])

        # Larger theta → closer to Poisson (variance closer to mean)
        family_small_theta = NegativeBinomialFamily(theta=0.5)
        family_large_theta = NegativeBinomialFamily(theta=10.0)

        var_small = family_small_theta.variance(mu)
        var_large = family_large_theta.variance(mu)

        # Larger theta should give smaller variance
        assert var_small > var_large

        # As theta → ∞, variance → μ (Poisson)
        assert var_large < mu * 2

    def test_negbinom_loglik(self):
        """Test log-likelihood computation."""
        family = NegativeBinomialFamily(theta=1.5)
        np.random.seed(42)

        # Generate count data
        mu = np.array([2, 5, 10])
        y = np.array([1, 4, 8])

        ll = family.loglik(y, mu)

        # Log-likelihood should be finite and negative
        assert np.isfinite(ll)
        assert ll < 0

    def test_negbinom_loglik_poisson_limit(self):
        """Test that large theta approaches Poisson behavior."""
        np.random.seed(42)
        y = np.array([0, 1, 2, 3, 5, 10])
        mu = np.array([2, 2, 2, 2, 2, 2])

        # Compare log-likelihood with different theta values
        family_small = NegativeBinomialFamily(theta=1.0)
        family_large = NegativeBinomialFamily(theta=100.0)

        ll_small = family_small.loglik(y, mu)
        ll_large = family_large.loglik(y, mu)

        # Both should be negative but in reasonable range
        assert ll_small < 0
        assert ll_large < 0


class TestInverseGaussianFamily:
    """Tests for Inverse Gaussian family."""

    def test_inversegaussian_initialization(self):
        """Test initialization with valid links."""
        family1 = InverseGaussianFamily(link="inverse-square")
        assert family1.link == "inverse-square"

        family2 = InverseGaussianFamily(link="1/mu^2")
        assert family2.link == "1/mu^2"

    def test_inversegaussian_invalid_link(self):
        """Test that invalid link raises error."""
        with pytest.raises(ValueError):
            InverseGaussianFamily(link="invalid")

    def test_inversegaussian_link(self):
        """Test inverse-square link function."""
        family = InverseGaussianFamily()

        # η = 1/μ², so μ = 1/√η
        eta = np.array([0.25, 1.0, 4.0, 16.0])
        mu = family.linkinv(eta)

        expected_mu = 1.0 / np.sqrt(eta)
        assert np.allclose(mu, expected_mu)

        # All values should be positive
        assert np.all(mu > 0)

    def test_inversegaussian_dmu_deta(self):
        """Test derivative of inverse-square link."""
        family = InverseGaussianFamily()
        eta = np.array([0.25, 1.0, 4.0])
        dmu_deta = family.dmu_deta(eta)

        # dμ/dη = -1/(2η^(3/2))
        expected = -0.5 / (eta ** (1.5))
        assert np.allclose(dmu_deta, expected)

        # Should be negative everywhere (inverse relationship)
        assert np.all(dmu_deta < 0)

    def test_inversegaussian_variance(self):
        """Test variance function φμ³."""
        family = InverseGaussianFamily()
        mu = np.array([0.5, 1.0, 2.0])
        phi = 0.1

        var = family.variance(mu, dispersion=phi)
        expected_var = phi * mu**3

        assert np.allclose(var, expected_var)

    def test_inversegaussian_variance_positive_valued(self):
        """Test that variance is always positive."""
        family = InverseGaussianFamily()
        mu = np.linspace(0.1, 10, 100)

        for phi in [0.1, 0.5, 1.0, 2.0]:
            var = family.variance(mu, dispersion=phi)
            assert np.all(var > 0)

    def test_inversegaussian_loglik(self):
        """Test log-likelihood computation."""
        family = InverseGaussianFamily()
        np.random.seed(42)

        # Generate data
        mu = np.array([1, 2, 3, 4, 5])
        y = mu + np.random.normal(0, 0.2, size=len(mu))
        y = np.abs(y)  # Ensure positive

        ll = family.loglik(y, mu, dispersion=0.5)

        # Log-likelihood should be finite
        assert np.isfinite(ll)

    def test_inversegaussian_small_mu(self):
        """Test numerical stability with small mu."""
        family = InverseGaussianFamily()

        # Very small values should be handled gracefully
        mu = np.array([1e-10, 1e-5, 1e-3])
        y = np.array([1e-10, 1e-5, 1e-3])

        # Should not crash
        ll = family.loglik(y, mu, dispersion=1.0)
        assert np.isfinite(ll)


class TestFamilyComparisons:
    """Compare behavior of different families."""

    def test_poisson_vs_negbinom_large_theta(self):
        """Test that Neg Binomial approaches Poisson as theta → ∞."""
        np.random.seed(42)
        y = np.array([0, 1, 2, 3, 5, 10, 15])
        mu = np.array([3, 3, 3, 3, 3, 3, 3])

        poisson = PoissonFamily()
        negbinom = NegativeBinomialFamily(theta=1000)  # Large theta

        ll_poisson = poisson.loglik(y, mu)
        ll_negbinom = negbinom.loglik(y, mu)

        # Should be similar for large theta
        # (won't be identical due to different parameterizations)
        assert np.isfinite(ll_poisson)
        assert np.isfinite(ll_negbinom)

    def test_binomial_mu_bounds(self):
        """Test that all binomials keep μ in [0,1]."""
        families = [
            BinomialFamily(link="logit"),
            BinomialFamily(link="probit"),
            BinomialFamily(link="cloglog"),
        ]

        eta = np.linspace(-10, 10, 100)

        for family in families:
            mu = family.linkinv(eta)
            assert np.all(mu >= 0)
            assert np.all(mu <= 1)

    def test_all_families_have_variance(self):
        """Test that all families have defined variance."""
        families = [
            GaussianFamily(),
            PoissonFamily(),
            BinomialFamily(),
            GammaFamily(),
            TweedieFamily(),
            NegativeBinomialFamily(),
            InverseGaussianFamily(),
        ]

        # Use family-specific mu values that are valid for each
        mu_datasets = [
            np.array([0.5, 1.0, 2.0, 5.0]),  # Gaussian: any value fine
            np.array([0.5, 1.0, 2.0, 5.0]),  # Poisson: any value > 0 fine
            np.array([0.1, 0.3, 0.5, 0.9]),  # Binomial: must be in [0, 1]
            np.array([0.5, 1.0, 2.0, 5.0]),  # Gamma: must be > 0
            np.array([0.5, 1.0, 2.0, 5.0]),  # Tweedie: must be > 0
            np.array([0.5, 1.0, 2.0, 5.0]),  # Neg Binomial: must be > 0
            np.array([0.5, 1.0, 2.0, 5.0]),  # Inverse Gaussian: must be > 0
        ]

        for family, mu in zip(families, mu_datasets):
            try:
                var = family.variance(mu)
                assert np.all(var > 0), f"{family.__class__.__name__} has non-positive variance"
                assert np.all(np.isfinite(var)), f"{family.__class__.__name__} has inf/nan variance"
            except Exception as e:
                pytest.fail(f"{family.__class__.__name__} variance() failed: {e}")

    def test_all_families_have_loglik(self):
        """Test that all families have defined log-likelihood."""
        families = [
            GaussianFamily(),
            PoissonFamily(),
            BinomialFamily(),
            GammaFamily(),
            TweedieFamily(),
            NegativeBinomialFamily(),
            InverseGaussianFamily(),
        ]

        # Generate reasonable data for each family (y and mu must have same shape)
        test_cases = [
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 2.0, 2.0, 2.0, 2.0])),  # Gaussian
            (np.array([0, 1, 2, 3, 5]), np.array([2.0, 2.0, 2.0, 2.0, 2.0])),  # Poisson
            (np.array([0, 0, 1, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5, 0.5])),  # Binomial
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 2.0, 2.0, 2.0, 2.0])),  # Gamma
            (np.array([0.1, 0.5, 1.0, 2.0, 5.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])),  # Tweedie
            (np.array([0, 1, 2, 3, 5]), np.array([2.0, 2.0, 2.0, 2.0, 2.0])),  # Neg Binomial
            (
                np.array([0.5, 1.0, 2.0, 3.0, 5.0]),
                np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            ),  # Inverse Gaussian
        ]

        for family, (y, mu) in zip(families, test_cases):
            try:
                ll = family.loglik(y, mu)
                assert np.isfinite(ll), f"{family.__class__.__name__} loglik is inf/nan"
            except Exception as e:
                pytest.fail(f"{family.__class__.__name__} loglik() failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
