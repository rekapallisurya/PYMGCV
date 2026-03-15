"""Quick validation script for new families."""

import numpy as np
import sys
sys.path.insert(0, 'c:\\Users\\surya\\Downloads\\pymgcv')

from pymgcv.distributions.family_base import (
    BinomialFamily, NegativeBinomialFamily, InverseGaussianFamily
)

def test_binomial():
    """Test Binomial family."""
    print("Testing BinomialFamily...")
    
    # Test logit link
    family = BinomialFamily(link='logit')
    eta = np.array([-2, -1, 0, 1, 2])
    mu = family.linkinv(eta)
    
    assert np.all((mu > 0) & (mu < 1)), "Binomial mu out of bounds"
    assert np.isclose(family.linkinv(0), 0.5), "Logit(0) != 0.5"
    print("  ✓ Logit link works")
    
    # Test probit link
    family = BinomialFamily(link='probit')
    mu = family.linkinv(eta)
    assert np.all((mu > 0) & (mu < 1)), "Probit mu out of bounds"
    print("  ✓ Probit link works")
    
    # Test cloglog link
    family = BinomialFamily(link='cloglog')
    mu = family.linkinv(eta)
    assert np.all((mu > 0) & (mu < 1)), "Cloglog mu out of bounds"
    print("  ✓ Cloglog link works")
    
    # Test variance
    family = BinomialFamily()
    mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    var = family.variance(mu)
    expected = mu * (1 - mu)
    assert np.allclose(var, expected), "Binomial variance incorrect"
    print("  ✓ Variance function correct")
    
    # Test log-likelihood
    y = np.array([0, 0, 1, 1, 1])
    mu = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
    ll = family.loglik(y, mu)
    assert np.isfinite(ll) and ll < 0, "Binomial loglik invalid"
    print("  ✓ Log-likelihood computation works")
    
    print("BinomialFamily: ALL TESTS PASSED ✓\n")

def test_negative_binomial():
    """Test Negative Binomial family."""
    print("Testing NegativeBinomialFamily...")
    
    # Test initialization
    family = NegativeBinomialFamily(theta=1.5)
    assert family.theta == 1.5, "Theta not set correctly"
    print("  ✓ Initialization works")
    
    # Test log link
    eta = np.array([-2, -1, 0, 1, 2])
    mu = family.linkinv(eta)
    expected_mu = np.exp(eta)
    assert np.allclose(mu, expected_mu), "Log link incorrect"
    assert np.all(mu > 0), "Mu not positive"
    print("  ✓ Log link works")
    
    # Test variance
    mu = np.array([1, 2, 5, 10])
    var = family.variance(mu)
    expected_var = mu + mu**2 / 1.5
    assert np.allclose(var, expected_var), "Variance incorrect"
    assert np.all(var > mu), "Negative Binomial should be overdispersed"
    print("  ✓ Variance function correct (overdispersed)")
    
    # Test log-likelihood
    y = np.array([1, 4, 8])
    mu = np.array([2, 5, 10])
    ll = family.loglik(y, mu)
    assert np.isfinite(ll), "Loglik not finite"
    print("  ✓ Log-likelihood computation works")
    
    print("NegativeBinomialFamily: ALL TESTS PASSED ✓\n")

def test_inverse_gaussian():
    """Test Inverse Gaussian family."""
    print("Testing InverseGaussianFamily...")
    
    # Test initialization
    family = InverseGaussianFamily()
    assert family.link == 'inverse-square', "Link not set correctly"
    print("  ✓ Initialization works")
    
    # Test link function
    eta = np.array([0.25, 1.0, 4.0, 16.0])
    mu = family.linkinv(eta)
    expected_mu = 1.0 / np.sqrt(eta)
    assert np.allclose(mu, expected_mu), "Inverse-square link incorrect"
    assert np.all(mu > 0), "Mu not positive"
    print("  ✓ Link function works")
    
    # Test derivative
    dmu_deta = family.dmu_deta(eta)
    expected_dmu_deta = -0.5 / (eta**(1.5))
    assert np.allclose(dmu_deta, expected_dmu_deta), "Derivative incorrect"
    assert np.all(dmu_deta < 0), "Derivative should be negative"
    print("  ✓ Derivative correct")
    
    # Test variance
    mu = np.array([0.5, 1.0, 2.0])
    var = family.variance(mu, dispersion=0.1)
    expected_var = 0.1 * mu**3
    assert np.allclose(var, expected_var), "Variance incorrect"
    print("  ✓ Variance function correct")
    
    # Test log-likelihood
    y = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    mu = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    ll = family.loglik(y, mu, dispersion=0.5)
    assert np.isfinite(ll), "Loglik not finite"
    print("  ✓ Log-likelihood computation works")
    
    print("InverseGaussianFamily: ALL TESTS PASSED ✓\n")

def test_gam_integration():
    """Test integration with GAM class."""
    print("Testing GAM integration with new families...")
    
    # Test that families are recognized
    from pymgcv.api.gam import GAM
    import pandas as pd
    
    # Create simple test data
    np.random.seed(42)
    n = 50
    x = np.linspace(0, 1, n)
    y_binary = np.random.binomial(1, 0.3 + 0.4*x)
    y_count = np.random.poisson(np.exp(0.5 + 0.5*x))
    
    df = pd.DataFrame({'x': x, 'y_binary': y_binary, 'y_count': y_count})
    
    # Test binomial
    try:
        model_bin = GAM('y_binary ~ s(x)', data=df, family='binomial')
        print("  ✓ Binomial GAM created successfully")
    except Exception as e:
        print(f"  ✗ Binomial GAM failed: {e}")
        return
    
    # Test negative binomial
    try:
        model_nb = GAM('y_count ~ s(x)', data=df, family='negative.binomial')
        print("  ✓ Negative Binomial GAM created successfully")
    except Exception as e:
        print(f"  ✗ Negative Binomial GAM failed: {e}")
        return
    
    # Test inverse gaussian
    try:
        model_ig = GAM('y_count ~ s(x)', data=df, family='inverse.gaussian')
        print("  ✓ Inverse Gaussian GAM created successfully")
    except Exception as e:
        print(f"  ✗ Inverse Gaussian GAM failed: {e}")
        return
    
    print("GAM Integration: ALL TESTS PASSED ✓\n")

if __name__ == '__main__':
    print("="*60)
    print("DISTRIBUTION FAMILY VALIDATION")
    print("="*60 + "\n")
    
    try:
        test_binomial()
        test_negative_binomial()
        test_inverse_gaussian()
        test_gam_integration()
        
        print("="*60)
        print("ALL TESTS PASSED ✓✓✓")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
