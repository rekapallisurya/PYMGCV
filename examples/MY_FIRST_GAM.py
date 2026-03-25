#!/usr/bin/env python
"""
PYMGCV STARTER TEMPLATE

Copy this file and modify it for your own GAM analysis.
This is the simplest way to get started with pymgcv.
"""

# Step 1: Import required packages
import numpy as np
import pandas as pd

from pymgcv.api.gam import GAM


# Step 2: Load or generate your data
def load_data():
    """Load or generate your data."""

    # OPTION A: Generate synthetic data
    np.random.seed(42)
    n = 150
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + 0.1 * x + np.random.normal(0, 0.3, n)

    # OPTION B: Load from CSV
    # data = pd.read_csv('your_data.csv')

    # Create DataFrame
    data = pd.DataFrame({"x": x, "y": y})

    return data


# Step 3: Fit the model
def fit_model(data):
    """Fit a GAM model."""

    # Create GAM object
    # Formula syntax: 'response ~ s(predictor1) + s(predictor2) + ...'
    model = GAM(
        formula="y ~ s(x, k=10)",
        family="gaussian",  # Can be: 'gaussian', 'poisson', 'binomial', 'gamma'
    )

    # Fit the model
    print("Fitting model...")
    model.fit(data, verbose=False)

    return model


# Step 4: Inspect results
def inspect_model(model):
    """Print model results."""

    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    # Print full summary if available
    if hasattr(model, "summary"):
        try:
            print(model.summary())
        except:
            pass

    # Extract key information
    print("\n--- Key Model Quantities ---")

    if hasattr(model, "coefficients"):
        print(f"Intercept: {model.coefficients[0]:.6f}")

    if hasattr(model, "edf"):
        print(f"EDF (Effective DF): {model.edf}")

    if hasattr(model, "lambda_"):
        print(f"Smoothing parameters: {model.lambda_}")

    if hasattr(model, "deviance"):
        print(f"Deviance: {model.deviance:.6f}")

    if hasattr(model, "aic"):
        print(f"AIC: {model.aic:.6f}")

    if hasattr(model, "gcv"):
        print(f"GCV: {model.gcv:.6f}")


# Step 5: Make predictions
def make_predictions(model, data):
    """Generate predictions."""

    # Create new data for prediction
    x_new = np.linspace(0, 2 * np.pi, 50)
    new_data = pd.DataFrame({"x": x_new})

    # Make predictions
    if hasattr(model, "predict"):
        predictions = model.predict(new_data)

        print("\n" + "=" * 70)
        print("PREDICTIONS")
        print("=" * 70)
        print(f"\n{'x':>8} | {'Prediction':>15}")
        print("-" * 25)
        for xi, pred in zip(x_new[::5], predictions[::5]):  # Show every 5th
            print(f"{xi:8.4f} | {pred:15.6f}")

        return new_data, predictions

    return None, None


# Step 6: Main execution
def main():
    """Run the analysis."""

    print("\n" + "#" * 70)
    print("# PYMGCV ANALYSIS")
    print("#" * 70)

    # Load data
    print("\n[1] Loading data...")
    data = load_data()
    print(f"    Data shape: {data.shape}")
    print(f"    Columns: {list(data.columns)}")

    # Fit model
    print("\n[2] Fitting GAM model...")
    model = fit_model(data)
    print("    ✓ Model fitted!")

    # Inspect results
    print("\n[3] Inspecting results...")
    inspect_model(model)

    # Make predictions
    print("\n[4] Making predictions...")
    new_data, predictions = make_predictions(model, data)

    print("\n" + "#" * 70)
    print("# ANALYSIS COMPLETE")
    print("#" * 70 + "\n")


# Run the analysis
if __name__ == "__main__":
    main()
