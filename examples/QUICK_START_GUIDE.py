#!/usr/bin/env python
"""
PYMGCV QUICK START GUIDE - Import & Basic Usage

This file demonstrates how to import and use pymgcv in your own Python programs.

Installation:
    pip install -e c:\Users\surya\Downloads\pymgcv
    (from the pymgcv directory)

Or add to your Python path:
    import sys
    sys.path.insert(0, 'c:/Users/surya/Downloads/pymgcv')
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

# Add pymgcv to path (if not installed via pip)
# sys.path.insert(0, 'c:/Users/surya/Downloads/pymgcv')


# ============================================================================
# EXAMPLE 1: BASIC IMPORT & SIMPLE GAUSSIAN GAM
# ============================================================================

def example_1_basic_gaussian():
    """Simplest example: Gaussian GAM with one smooth term."""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Gaussian GAM")
    print("="*80)
    
    try:
        from pymgcv.api.gam import GAM
        
        # Step 1: Create data
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 1, n)
        y = np.sin(6*np.pi*x) + np.random.normal(0, 0.1, n)
        
        data = pd.DataFrame({'x': x, 'y': y})
        
        # Step 2: Create GAM model
        model = GAM('y ~ s(x)', family='gaussian')
        
        # Step 3: Fit the model
        print("\nFitting model...")
        model.fit(data, verbose=False)
        
        # Step 4: Print results
        print("\n✓ Model fitted successfully!")
        
        if hasattr(model, 'summary'):
            print(model.summary())
        
        # Step 5: Get coefficients
        if hasattr(model, 'coefficients'):
            print(f"\nIntercept: {model.coefficients[0]:.6f}")
        
        # Step 6: Make predictions
        x_new = np.array([0.1, 0.5, 0.9])
        data_new = pd.DataFrame({'x': x_new})
        if hasattr(model, 'predict'):
            predictions = model.predict(data_new)
            print(f"\nPredictions at x={x_new}:")
            for xi, pred in zip(x_new, predictions):
                print(f"  x={xi:.1f}: {pred:.6f}")
        
        return model
        
    except ImportError as e:
        print(f"✗ Import Error: {e}")
        print("  Make sure pymgcv is installed or added to Python path")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# EXAMPLE 2: MULTIPLE SMOOTH TERMS
# ============================================================================

def example_2_multiple_smooths():
    """Fit a GAM with multiple smooth terms."""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Multiple Smooth Terms")
    print("="*80)
    
    try:
        from pymgcv.api.gam import GAM
        
        # Generate data
        np.random.seed(123)
        n = 150
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(0, 2, n)
        y = (np.sin(6*np.pi*x1) + 
             np.cos(4*np.pi*x2) + 
             np.random.normal(0, 0.2, n))
        
        data = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'y': y
        })
        
        # Fit model with two smooth terms
        model = GAM('y ~ s(x1) + s(x2)', family='gaussian')
        
        print("\nFitting model with two smooth terms...")
        model.fit(data, verbose=False)
        
        print("✓ Model fitted successfully!")
        
        if hasattr(model, 'edf'):
            print(f"\nEDF (Effective Degrees of Freedom):")
            for term, edf in model.edf.items():
                print(f"  {term}: {edf:.2f}")
        
        return model
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# EXAMPLE 3: POISSON GLM (COUNT DATA)
# ============================================================================

def example_3_poisson_glm():
    """Fit a Poisson GAM for count data."""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Poisson GAM (Count Data)")
    print("="*80)
    
    try:
        from pymgcv.api.gam import GAM
        
        # Generate count data
        np.random.seed(456)
        n = 120
        x = np.linspace(0, 2, n)
        
        # True rate: lambda = exp(0.5 + 1.2*x - 0.3*x^2)
        eta = 0.5 + 1.2*x - 0.3*x**2
        mu = np.exp(eta)
        y = np.random.poisson(mu)
        
        data = pd.DataFrame({'x': x, 'y': y})
        
        # Fit Poisson GAM
        model = GAM('y ~ s(x, k=12)', family='poisson')
        
        print("\nFitting Poisson model...")
        model.fit(data, verbose=False)
        
        print("✓ Poisson model fitted!")
        
        if hasattr(model, 'deviance'):
            print(f"Deviance: {model.deviance:.4f}")
        
        if hasattr(model, 'coefficients'):
            print(f"Intercept (link scale): {model.coefficients[0]:.4f}")
        
        return model
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# EXAMPLE 4: WITH PARAMETRIC TERMS
# ============================================================================

def example_4_mixed_model():
    """Fit a GAM with both parametric and smooth terms."""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Mixed Parametric & Smooth Terms")
    print("="*80)
    
    try:
        from pymgcv.api.gam import GAM
        
        # Generate data
        np.random.seed(789)
        n = 100
        x = np.linspace(0, 1, n)
        z = np.random.normal(0, 1, n)
        y = (0.5 + 2*z + 
             np.sin(6*np.pi*x) + 
             np.random.normal(0, 0.15, n))
        
        data = pd.DataFrame({
            'x': x,
            'z': z,
            'y': y
        })
        
        # Model: y ~ parametric_z + smooth_x
        model = GAM('y ~ z + s(x)', family='gaussian')
        
        print("\nFitting mixed model...")
        model.fit(data, verbose=False)
        
        print("✓ Model fitted!")
        
        if hasattr(model, 'coefficients'):
            print(f"\nParametric coefficient (z): {model.coefficients[1]:.4f}")
            print(f"Intercept: {model.coefficients[0]:.4f}")
        
        return model
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# EXAMPLE 5: CUSTOM SETTINGS & CONVERGENCE
# ============================================================================

def example_5_advanced_settings():
    """GAM with custom settings."""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Settings & Options")
    print("="*80)
    
    try:
        from pymgcv.api.gam import GAM
        
        # Generate data
        np.random.seed(999)
        n = 200
        x = np.linspace(0, 2*np.pi, n)
        y = (np.sin(x) + 0.1*x + 
             np.random.normal(0, 0.25, n))
        
        data = pd.DataFrame({'x': x, 'y': y})
        
        # Create model with custom settings
        model = GAM(
            formula='y ~ s(x, k=15)',
            family='gaussian',
            verbose=True  # Show convergence info
        )
        
        print("\nFitting with verbose output...")
        model.fit(data, 
                 max_iter=100,      # Maximum PIRLS iterations
                 tol=1e-7)          # Convergence tolerance
        
        print("\n✓ Model fitted with custom settings!")
        
        return model
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# EXAMPLE 6: PREDICTIONS & CONFIDENCE INTERVALS
# ============================================================================

def example_6_predictions_with_ci():
    """Make predictions with confidence intervals."""
    
    print("\n" + "="*80)
    print("EXAMPLE 6: Predictions & Confidence Intervals")
    print("="*80)
    
    try:
        from pymgcv.api.gam import GAM
        
        # Generate data
        np.random.seed(111)
        n = 100
        x = np.linspace(0, 1, n)
        y = (3 + 2*x + np.sin(4*np.pi*x) + 
             np.random.normal(0, 0.2, n))
        
        data = pd.DataFrame({'x': x, 'y': y})
        
        # Fit model
        model = GAM('y ~ s(x, k=12)')
        model.fit(data, verbose=False)
        
        # Make predictions on new data
        x_new = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        data_new = pd.DataFrame({'x': x_new})
        
        print("\nMaking predictions...")
        
        if hasattr(model, 'predict'):
            predictions = model.predict(data_new)
            
            print(f"\n{'x':>6} | {'Prediction':>12}")
            print("-" * 20)
            for xi, pred in zip(x_new, predictions):
                print(f"{xi:6.2f} | {pred:12.6f}")
        
        # Try to get confidence intervals (if implemented)
        if hasattr(model, 'predict_with_ci'):
            print("\n(Confidence intervals available via model.predict_with_ci())")
        
        return model
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# EXAMPLE 7: EXTRACT MODEL INFO
# ============================================================================

def example_7_model_inspection():
    """Extract and inspect model information."""
    
    print("\n" + "="*80)
    print("EXAMPLE 7: Model Inspection & Diagnostics")
    print("="*80)
    
    try:
        from pymgcv.api.gam import GAM
        
        # Generate data
        np.random.seed(222)
        n = 150
        x = np.linspace(0, 2, n)
        y = (np.exp(0.1*x) + np.sin(3*x) + 
             np.random.normal(0, 0.3, n))
        
        data = pd.DataFrame({'x': x, 'y': y})
        
        # Fit model
        model = GAM('y ~ s(x)')
        model.fit(data, verbose=False)
        
        print("\n--- Model Attributes ---")
        
        # Coefficients
        if hasattr(model, 'coefficients'):
            print(f"Coefficients shape: {model.coefficients.shape}")
            print(f"Coefficients: {model.coefficients}")
        
        # Standard errors
        if hasattr(model, 'se'):
            print(f"\nStandard errors: {model.se}")
        
        # EDF (Effective Degrees of Freedom)
        if hasattr(model, 'edf'):
            print(f"\nEDF: {model.edf}")
        
        # Smoothing parameters
        if hasattr(model, 'lambda_'):
            print(f"Smoothing parameters: {model.lambda_}")
        
        # Deviance
        if hasattr(model, 'deviance'):
            print(f"Deviance: {model.deviance:.6f}")
        
        # AIC
        if hasattr(model, 'aic'):
            if np.isfinite(model.aic):
                print(f"AIC: {model.aic:.6f}")
            else:
                print(f"AIC: Not available")
        
        # GCV
        if hasattr(model, 'gcv'):
            if np.isfinite(model.gcv):
                print(f"GCV: {model.gcv:.6f}")
            else:
                print(f"GCV: Not available")
        
        # Formula
        if hasattr(model, 'formula'):
            print(f"\nFormula: {model.formula}")
        
        # Family
        if hasattr(model, 'family'):
            print(f"Family: {model.family}")
        
        return model
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# MAIN: RUN ALL EXAMPLES
# ============================================================================

def main():
    """Run all examples."""
    
    print("\n" + "#"*80)
    print("# PYMGCV QUICK START EXAMPLES")
    print("#"*80)
    
    print("\nTo use pymgcv in your code:")
    print("  1. from pymgcv.api.gam import GAM")
    print("  2. Create data as pandas DataFrame")
    print("  3. model = GAM('formula string')")
    print("  4. model.fit(data)")
    print("  5. model.summary() or model.predict()")
    
    # Run examples
    models = {}
    models['example1'] = example_1_basic_gaussian()
    models['example2'] = example_2_multiple_smooths()
    models['example3'] = example_3_poisson_glm()
    models['example4'] = example_4_mixed_model()
    models['example5'] = example_5_advanced_settings()
    models['example6'] = example_6_predictions_with_ci()
    models['example7'] = example_7_model_inspection()
    
    # Summary
    print("\n" + "#"*80)
    print("# SUMMARY")
    print("#"*80)
    successful = sum(1 for m in models.values() if m is not None)
    print(f"\n✓ {successful}/{len(models)} examples completed successfully")
    
    print("\n" + "#"*80)
    print("# NEXT STEPS")
    print("#"*80)
    print("""
1. Create your own Python file
2. Add these imports:
   
   from pymgcv.api.gam import GAM
   import pandas as pd
   import numpy as np

3. Load or generate your data as a pandas DataFrame
4. Create a model: model = GAM('y ~ s(x)')
5. Fit: model.fit(data)
6. Inspect: print(model.summary())
7. Predict: predictions = model.predict(new_data)

See examples above for detailed patterns!
""")


if __name__ == '__main__':
    main()
