#!/usr/bin/env python
"""Comprehensive pymgcv vs mgcv output comparison template.

This script generates pymgcv output in a format suitable for comparison with R's mgcv.

USAGE:
    1. Run this script to generate pymgcv output
    2. Run equivalent R code (provided below)
    3. Compare outputs side-by-side

EQUIVALENT R CODE:
    library(mgcv)

    # Generate data
    set.seed(42)
    n <- 150
    x <- seq(0, 2*pi, length.out=n)
    y <- sin(x) + 0.1*x + rnorm(n, 0, 0.3)

    # Fit GAM
    fit_gam <- gam(y ~ s(x, k=10), family=gaussian())

    # Summary
    summary(fit_gam)

    # Predictions
    x_pred <- seq(0, 2*pi, length.out=50)
    pred <- predict(fit_gam, newdata=data.frame(x=x_pred), se.fit=TRUE)

    # AIC, GCV
    AIC(fit_gam)
    fit_gam$gcv.ubre

    # Coefficients extract
    coef(fit_gam)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_pymgcv_example() -> dict:
    """Run pymgcv example and return results as dict."""
    print("\n" + "=" * 80)
    print("PYMGCV EXAMPLE: GAUSSIAN GAM WITH SMOOTH TERM")
    print("=" * 80)

    # Generate data
    print("\n[STEP 1] Data Generation")
    print("-" * 80)
    np.random.seed(42)
    n = 150
    x = np.linspace(0, 2 * np.pi, n)
    f_true = np.sin(x) + 0.1 * x
    noise = np.random.normal(0, 0.3, n)
    y = f_true + noise

    data = pd.DataFrame({"x": x, "y": y})

    print(f"Generated {n} observations")
    print(f"  x range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Data shape: {data.shape}")

    # Try to fit model
    print("\n[STEP 2] Model Fitting")
    print("-" * 80)

    results = {
        "formula": "y ~ s(x, k=10)",
        "family": "gaussian",
        "observations": n,
        "data": data.to_dict("list"),  # Store as JSON-serializable
        "model_fitted": False,
        "errors": [],
    }

    try:
        from pymgcv.api.gam import GAM

        print("Creating GAM object with formula: y ~ s(x, k=10)")
        model = GAM("y ~ s(x, k=10)", family="gaussian")

        print("Fitting model...")
        model.fit(data, verbose=False)

        results["model_fitted"] = True
        print("✓ Model fitted successfully!")

        # Extract components
        print("\n[STEP 3] Extracting Model Components")
        print("-" * 80)

        # Coefficients
        if hasattr(model, "coefficients"):
            coef = model.coefficients
            results["coefficients"] = {
                "intercept": float(coef[0]) if len(coef) > 0 else None,
                "raw": coef.tolist() if hasattr(coef, "tolist") else None,
            }
            print(f"Coefficients extracted: {len(coef)} parameters")

        # EDF
        if hasattr(model, "edf"):
            edf_dict = model.edf
            results["edf"] = {k: float(v) for k, v in edf_dict.items()}
            print(f"EDF per term: {results['edf']}")

        # Smoothing parameters
        if hasattr(model, "lambda_"):
            lam = model.lambda_
            results["smoothing_param"] = {k: float(v) for k, v in lam.items()}
            print(f"Smoothing parameters: {results['smoothing_param']}")

        # Predictions
        print("\n[STEP 4] Generating Predictions")
        print("-" * 80)
        x_pred = np.linspace(0, 2 * np.pi, 50)
        data_pred = pd.DataFrame({"x": x_pred})

        if hasattr(model, "predict"):
            pred = model.predict(data_pred)
            results["predictions"] = {
                "x": x_pred.tolist(),
                "fitted": pred.tolist() if hasattr(pred, "tolist") else None,
            }
            print(f"Generated {len(pred)} predictions")
            print(f"Prediction range: [{min(pred):.4f}, {max(pred):.4f}]")

        # Model statistics
        print("\n[STEP 5] Model Statistics")
        print("-" * 80)

        if hasattr(model, "deviance"):
            dev = model.deviance
            results["deviance"] = float(dev)
            print(f"Deviance: {dev:.6f}")

        if hasattr(model, "aic"):
            aic = model.aic
            results["aic"] = float(aic) if np.isfinite(aic) else None
            print(f"AIC: {aic:.6f}" if np.isfinite(aic) else "AIC: Not computed")

        if hasattr(model, "gcv"):
            gcv = model.gcv
            results["gcv"] = float(gcv) if np.isfinite(gcv) else None
            print(f"GCV: {gcv:.6f}" if np.isfinite(gcv) else "GCV: Not computed")

        # Standard errors
        if hasattr(model, "se"):
            se = model.se
            results["standard_errors"] = {
                "intercept": float(se[0]) if len(se) > 0 else None,
                "raw": se.tolist() if hasattr(se, "tolist") else None,
            }
            print(f"Standard errors extracted: {len(se)} parameters")

    except Exception as e:
        results["errors"].append(str(e))
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()

    return results


def print_comparison_template(pymgcv_results: dict) -> str:
    """Print template for side-by-side comparison with R mgcv."""
    template = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                    PYMGCV vs MGCV OUTPUT COMPARISON                            ║
╚════════════════════════════════════════════════════════════════════════════════╝

PYMGCV RESULTS
──────────────────────────────────────────────────────────────────────────────────
Formula: {formula}
Family:  {family}
Observations: {observations}

Model Status: {"FITTED" if pymgcv_results.get('model_fitted') else "FAILED"}
"""

    if pymgcv_results.get("errors"):
        template += "\nErrors:\n"
        for err in pymgcv_results["errors"]:
            template += f"  - {err}\n"

    if pymgcv_results.get("coefficients"):
        template += "\nCoefficients:\n"
        coef = pymgcv_results["coefficients"]
        if coef.get("intercept") is not None:
            template += f"  Intercept: {coef['intercept']:.6f}\n"

    if pymgcv_results.get("edf"):
        template += "\nEffective Degrees of Freedom (EDF):\n"
        for term, edf_val in pymgcv_results["edf"].items():
            template += f"  {term}: {edf_val:.4f}\n"

    if pymgcv_results.get("smoothing_param"):
        template += "\nSmoothing Parameters (λ):\n"
        for term, lam in pymgcv_results["smoothing_param"].items():
            template += f"  {term}: {lam:.6e}\n"

    if pymgcv_results.get("aic") is not None:
        template += f"\nAIC: {pymgcv_results['aic']:.6f}\n"

    if pymgcv_results.get("gcv") is not None:
        template += f"GCV: {pymgcv_results['gcv']:.6f}\n"

    if pymgcv_results.get("deviance") is not None:
        template += f"Deviance: {pymgcv_results['deviance']:.6f}\n"

    template += """

R MGCV EQUIVALENT CODE
──────────────────────────────────────────────────────────────────────────────────
library(mgcv)
set.seed(42)
n <- 150
x <- seq(0, 2*pi, length.out=n)
y <- sin(x) + 0.1*x + rnorm(n, 0, 0.3)
fit <- gam(y ~ s(x, k=10), family=gaussian())
summary(fit)
AIC(fit)
coef(fit)

[Paste R output below for comparison]


EXPECTED EQUIVALENCES
──────────────────────────────────────────────────────────────────────────────────
✓ Coefficients should agree within 1e-6
✓ EDF should be within 0.01
✓ Smoothing parameters may differ in scale but should show similar fit quality
✓ AIC values should be identical
✓ GCV scores should match


NOTES FOR R COMPARISON
──────────────────────────────────────────────────────────────────────────────────
1. PyMGCV uses identical statistical algorithms as mgcv
2. Numerical precision may differ in the 6-8 decimal places
3. Smoothing parameter optimization may converge to same value differently
4. GCV and REML scores should achieve same minimum
"""

    return template.format(**pymgcv_results)


def main() -> None:
    """Main entry point."""
    # Run pymgcv example
    results = run_pymgcv_example()

    # Print comparison template
    template = print_comparison_template(results)
    print(template)

    # Save results to JSON for inspection
    output_file = Path("pymgcv_output.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
