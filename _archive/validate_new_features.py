"""
Simple validation of new distribution families in pymgcv
Testing against realistic GAM scenarios.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'c:\\Users\\surya\\Downloads\\pymgcv')

from pymgcv.api.gam import GAM


def test_binomial():
    """Test Binomial GAM"""
    print("\n[TEST 1] Binomial GAM (Binary Classification)")
    print("-" * 50)
    
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    p_true = 0.5 + 0.3 * np.sin(6 * np.pi * x)
    p_true = np.clip(p_true, 0.01, 0.99)
    y = np.random.binomial(1, p_true)
    
    df = pd.DataFrame({'x': x, 'y': y})
    print(f"Data: {n} binary observations")
    print(f"Response rate: {y.mean():.1%}")
    
    try:
        model = GAM('y ~ s(x)', family='binomial')
        model.fit(df, max_outer_iter=3, verbose=False)
        print("[PASS] Binomial GAM fitted successfully")
        
        # Test prediction
        pred = model.predict(pd.DataFrame({'x': [0.5]}))
        if np.isfinite(pred).all():
            print(f"[PASS] Predictions valid (sample: {pred[0]:.4f})")
        else:
            print("[FAIL] Predictions contain NaN/Inf")
    except Exception as e:
        print(f"[FAIL] {e}")


def test_negative_binomial():
    """Test Negative Binomial GAM"""
    print("\n[TEST 2] Negative Binomial GAM (Count Data)")
    print("-" * 50)
    
    np.random.seed(42)
    n = 80
    x = np.linspace(0, 4, n)
    eta_true = 1.2 + 0.5*x
    mu_true = np.exp(eta_true)
    y = np.random.negative_binomial(n=2, p=2/(2+mu_true))
    
    df = pd.DataFrame({'x': x, 'y': y})
    print(f"Data: {n} count observations")
    print(f"Mean: {y.mean():.2f}, Variance: {y.var():.2f}")
    print(f"Overdispersion: {y.var()/y.mean():.2f}")
    
    try:
        model = GAM('y ~ s(x)', family='negative.binomial')
        model.fit(df, max_outer_iter=3, verbose=False)
        print("[PASS] Negative Binomial GAM fitted")
        
        # Test prediction
        pred = model.predict(pd.DataFrame({'x': [2.0]}))
        if np.isfinite(pred).all() and np.all(pred > 0):
            print(f"[PASS] Predictions valid (sample: {pred[0]:.4f})")
        else:
            print("[FAIL] Predictions invalid")
    except Exception as e:
        print(f"[FAIL] {e}")


def test_inverse_gaussian():
    """Test Inverse Gaussian GAM"""
    print("\n[TEST 3] Inverse Gaussian GAM (Heavy-tailed)")
    print("-" * 50)
    
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    mu_true = np.exp(1.5 + 1.2*x)
    y = np.random.exponential(scale=mu_true)
    
    df = pd.DataFrame({'x': x, 'y': y})
    print(f"Data: {n} positive continuous observations")
    print(f"Mean: {y.mean():.2f}, Std: {y.std():.2f}")
    print(f"Min: {y.min():.2f}, Max: {y.max():.2f}")
    
    try:
        model = GAM('y ~ s(x)', family='inverse.gaussian')
        model.fit(df, max_outer_iter=3, verbose=False)
        print("[PASS] Inverse Gaussian GAM fitted")
        
        # Test prediction
        pred = model.predict(pd.DataFrame({'x': [0.5]}))
        if np.isfinite(pred).all() and np.all(pred > 0):
            print(f"[PASS] Predictions valid (sample: {pred[0]:.4f})")
        else:
            print("[FAIL] Predictions invalid")
    except Exception as e:
        print(f"[FAIL] {e}")


def test_gaussian():
    """Test that Gaussian still works"""
    print("\n[TEST 4] Gaussian GAM (Baseline)")
    print("-" * 50)
    
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = 2 + 3*x + 2*np.sin(4*np.pi*x) + np.random.normal(0, 0.3, n)
    
    df = pd.DataFrame({'x': x, 'y': y})
    print(f"Data: {n} continuous observations")
    print(f"Mean: {y.mean():.2f}, Std: {y.std():.2f}")
    
    try:
        model = GAM('y ~ s(x)', family='gaussian')
        model.fit(df, max_outer_iter=3, verbose=False)
        print("[PASS] Gaussian GAM fitted")
        
        # Test prediction
        pred = model.predict(pd.DataFrame({'x': [0.5]}))
        if np.isfinite(pred).all():
            print(f"[PASS] Predictions valid (sample: {pred[0]:.4f})")
        else:
            print("[FAIL] Predictions invalid")
    except Exception as e:
        print(f"[FAIL] {e}")


def test_poisson():
    """Test that Poisson still works"""
    print("\n[TEST 5] Poisson GAM (Baseline)")
    print("-" * 50)
    
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 3, n)
    eta_true = 0.5 + 0.6*x
    mu_true = np.exp(eta_true)
    y = np.random.poisson(mu_true)
    
    df = pd.DataFrame({'x': x, 'y': y})
    print(f"Data: {n} count observations")
    print(f"Mean: {y.mean():.2f}, Variance: {y.var():.2f}")
    
    try:
        model = GAM('y ~ s(x)', family='poisson')
        model.fit(df, max_outer_iter=3, verbose=False)
        print("[PASS] Poisson GAM fitted")
        
        # Test prediction
        pred = model.predict(pd.DataFrame({'x': [1.5]}))
        if np.isfinite(pred).all() and np.all(pred > 0):
            print(f"[PASS] Predictions valid (sample: {pred[0]:.4f})")
        else:
            print("[FAIL] Predictions invalid")
    except Exception as e:
        print(f"[FAIL] {e}")


def test_summary_attributes():
    """Test model summary attributes"""
    print("\n[TEST 6] Model Attributes & Summary")
    print("-" * 50)
    
    np.random.seed(42)
    n = 50
    x = np.linspace(0, 1, n)
    y = 2 + x + np.random.normal(0, 0.2, n)
    df = pd.DataFrame({'x': x, 'y': y})
    
    try:
        model = GAM('y ~ s(x)', family='gaussian')
        model.fit(df, max_outer_iter=2)
        
        # Check attributes
        attrs_found = []
        attrs_missing = []
        
        for attr in ['beta', 'edf', 'fitted', 'family']:
            if hasattr(model, attr):
                attrs_found.append(attr)
            else:
                attrs_missing.append(attr)
        
        if attrs_found:
            print(f"[PASS] Found attributes: {', '.join(attrs_found)}")
        if attrs_missing:
            print(f"[WARN] Missing attributes: {', '.join(attrs_missing)}")
        
        # Try summary
        try:
            summary_text = model.summary()
            if isinstance(summary_text, str) and len(summary_text) > 0:
                print("[PASS] Summary generates output")
            else:
                print("[WARN] Summary output empty")
        except:
            print("[WARN] Summary method not available")
            
    except Exception as e:
        print(f"[FAIL] {e}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("PYMGCV DISTRIBUTION FAMILY VALIDATION")
    print("="*70)
    
    try:
        test_binomial()
        test_negative_binomial()
        test_inverse_gaussian()
        test_gaussian()
        test_poisson()
        test_summary_attributes()
        
        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED by user]")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
