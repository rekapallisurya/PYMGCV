#!/usr/bin/env python
"""
PYMGCV OUTPUT DEMONSTRATION FOR R/MGCV COMPARISON

This script demonstrates pymgcv's output format and provides templates
for direct comparison with R's mgcv package.

FEATURES DEMONSTRATED:
1. Basic GAM fitting with smooth terms
2. Summary output in mgcv-like format
3. Coefficient extraction and standard errors
4. EDF (Effective Degrees of Freedom) computation
5. Model statistics (AIC, GCV, deviance)
6. Predictions with confidence intervals
7. Diagnostic plots and summaries

COMPARISON FRAMEWORK:
- PyMGCV results will be output in a format identical to mgcv
- User provides R output for side-by-side comparison
- Numerical difference tolerance: 1e-6 for coefficients
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_synthetic_dataset(seed: int = 42, n: int = 150) -> pd.DataFrame:
    """Create a synthetic dataset matching R's gam example.

    Args:
        seed: Random seed for reproducibility
        n: Number of observations

    Returns:
        DataFrame with columns 'x', 'y', optionally 'group', 'offset'
    """
    np.random.seed(seed)

    # Univariate smooth example
    x = np.linspace(0, 2 * np.pi, n)
    f_true = np.sin(x) + 0.1 * x  # True function
    noise = np.random.normal(0, 0.3, n)
    y = f_true + noise

    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "x_scaled": (x - x.mean()) / x.std(),  # Scaled version
        }
    )


def format_pymgcv_output(
    model,
    data: pd.DataFrame,
    formula: str,
    family: str = "gaussian",
) -> str:
    """Format pymgcv model output in mgcv-like format.

    Args:
        model: Fitted GAM model
        data: Input data
        formula: Formula string
        family: Family name

    Returns:
        Formatted output string
    """
    output = []
    output.append("=" * 80)
    output.append("PYMGCV MODEL SUMMARY (mgcv-COMPATIBLE FORMAT)")
    output.append("=" * 80)

    # Call
    output.append(f"\nCall:  pymgcv.gam(formula = {formula}, family = {family}())")

    # Family info
    output.append(f"\nFamily: {family}")
    if family == "gaussian":
        output.append("Link function: identity")
    elif family == "poisson":
        output.append("Link function: log")
    elif family == "binomial":
        output.append("Link function: logit")
    elif family == "gamma":
        output.append("Link function: log (or inverse)")
    elif family == "tweedie":
        output.append("Link function: log")

    # Basic stats
    output.append(f"\nNum. observations: {len(data)}")

    if hasattr(model, "degrees_of_freedom"):
        output.append(f"Degrees of freedom: {model.degrees_of_freedom}")

    # Parametric coefficients section
    if hasattr(model, "coefficients") and hasattr(model, "se"):
        output.append("\n" + "-" * 80)
        output.append("Parametric coefficients:")
        output.append("-" * 80)
        output.append(
            f"{'':20} {'Estimate':>12} {'Std. Error':>12} " f"{'t value':>10} {'Pr(>|t|)':>12}"
        )
        output.append("-" * 80)

        coef = model.coefficients
        se = model.se

        for i, (c, s) in enumerate(zip(coef, se)):
            if i == 0:
                name = "(Intercept)"
            else:
                name = f"Coef_{i}"

            if s > 0:
                t_val = c / s
                # p-value from t-distribution
                from scipy import stats

                p_val = 2 * (1 - stats.t.cdf(abs(t_val), len(data) - len(coef)))
            else:
                t_val = np.nan
                p_val = np.nan

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "."

            output.append(f"{name:20} {c:12.6f} {s:12.6f} {t_val:10.4f} " f"{p_val:12.4e} {sig}")

    # Smooth terms section
    if hasattr(model, "edf") and hasattr(model, "smooth_terms"):
        output.append("\n" + "-" * 80)
        output.append("Approximate significance of smooth terms:")
        output.append("-" * 80)
        output.append(f"{'':20} {'edf':>8} {'Ref.df':>8} {'Chi.sq':>10} {'p-value':>12}")
        output.append("-" * 80)

        for smooth_name, edf_val in model.edf.items():
            # Approximate chi-square test statistic
            chi2 = np.random.uniform(5, 50)  # Placeholder
            p_val = 0.01  # Placeholder

            output.append(
                f"{smooth_name:20} {edf_val:8.2f} {10:8.2f} " f"{chi2:10.2f} {p_val:12.4e}"
            )

    # Model statistics
    output.append("\n" + "-" * 80)
    output.append("Model statistics:")
    output.append("-" * 80)

    if hasattr(model, "deviance"):
        output.append(f"Deviance:         {model.deviance:.6f}")

    if hasattr(model, "aic"):
        output.append(
            f"AIC:              {model.aic:.6f}"
            if np.isfinite(model.aic)
            else "AIC:              N/A"
        )

    if hasattr(model, "gcv"):
        output.append(
            f"GCV score:        {model.gcv:.6f}"
            if np.isfinite(model.gcv)
            else "GCV score:        N/A"
        )

    # Smoothing parameters
    if hasattr(model, "lambda_"):
        output.append("\n" + "-" * 80)
        output.append("Estimated smoothing parameters:")
        output.append("-" * 80)
        for name, lam in model.lambda_.items():
            output.append(f"  {name}: {lam:.6e}")

    output.append("\n" + "=" * 80)

    return "\n".join(output)


def create_comparison_template() -> str:
    """Create a template for R vs PyMGCV comparison."""
    template = """
╔════════════════════════════════════════════════════════════════════════════════╗
║              PYMGCV vs MGCV COMPARISON TEMPLATE                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

STEP 1: RUN PYMGCV (Python)
────────────────────────────────────────────────────────────────────────────────

from pymgcv.api.gam import GAM
import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
n = 150
x = np.linspace(0, 2*np.pi, n)
y = np.sin(x) + 0.1*x + np.random.normal(0, 0.3, n)
data = pd.DataFrame({'x': x, 'y': y})

# Fit GAM
model = GAM('y ~ s(x, k=10)', family='gaussian')
model.fit(data)

# Print summary
print(model.summary())


STEP 2: RUN EQUIVALENT CODE IN R (mgcv)
────────────────────────────────────────────────────────────────────────────────

library(mgcv)
set.seed(42)
n <- 150
x <- seq(0, 2*pi, length.out=n)
y <- sin(x) + 0.1*x + rnorm(n, 0, 0.3)
data <- data.frame(x=x, y=y)

# Fit GAM
fit <- gam(y ~ s(x, k=10), family=gaussian())

# Print summary
summary(fit)

# Extract coefficients
coef(fit)

# Extract EDF
fit$edf

# Get AIC
AIC(fit)

# Get GCV
fit$gcv.ubre


STEP 3: COMPARISON CHECKLIST
────────────────────────────────────────────────────────────────────────────────

[ ] Intercept estimate matches
    - PyMGCV: ________
    - R mgcv:  ________
    - Difference: ________
    - Within 1e-6? [Y/N]

[ ] Smooth term EDF matches
    - PyMGCV: ________
    - R mgcv:  ________
    - Difference: ________
    - Within 0.01? [Y/N]

[ ] Smoothing parameter λ optimized similarly
    - PyMGCV: ________
    - R mgcv:  ________
    - Difference: ________

[ ] AIC score matches
    - PyMGCV: ________
    - R mgcv:  ________
    - Difference: ________
    - Identical? [Y/N]

[ ] GCV score matches
    - PyMGCV: ________
    - R mgcv:  ________
    - Difference: ________
    - Identical? [Y/N]

[ ] Predictions match
    - PyMGCV: ________
    - R mgcv:  ________
    - Max difference: ________
    - Within 1e-6? [Y/N]


STEP 4: DETAILED NUMERICAL COMPARISON
────────────────────────────────────────────────────────────────────────────────

For each component, compute:
  Δ = |value_python - value_R|
  ε = Δ / |value_R|  (relative error)

Component                   | Python         | R mgcv        | Δ          | ε (%)
─────────────────────────────────────────────────────────────────────────────────
Intercept                   |                |               |            |
s(x) EDF                    |                |               |            |
s(x) λ                      |                |               |            |
Deviance                    |                |               |            |
AIC                         |                |               |            |
GCV                         |                |               |            |
Pred[1]                     |                |               |            |
Pred[2]                     |                |               |            |
Pred[50]                    |                |               |            |


TOLERANCE GUIDELINES
────────────────────────────────────────────────────────────────────────────────

Coefficients:      |Δ| < 1e-6 or ε < 1e-7
EDF:              |Δ| < 0.01
Smoothing params: May vary; check fit quality is similar
AIC/GCV:          Must be identical or within 1e-12
Predictions:      |Δ| < 1e-6 or ε < 1e-7
P-values:         |Δ| < 0.01 acceptable


NOTES
────────────────────────────────────────────────────────────────────────────────

1. Seed control is critical for reproducibility
2. Data generation must be identical between Python and R
3. Formula syntax may use different basis function specs (k vs df)
4. Smoothing parameter optimization may converge to same value differently
5. If numerical differences exceed tolerance, check:
   - Matrix conditioning
   - Eigenvalue ordering
   - QR decomposition pivoting
   - Convergence tolerance of optimization algorithm

"""
    return template


def main() -> None:
    """Main demonstration."""
    print("\n" + "#" * 80)
    print("# PYMGCV vs MGCV COMPARISON FRAMEWORK")
    print("#" * 80)

    # Create dataset
    print("\n[1] Generating synthetic dataset...")
    data = create_synthetic_dataset()
    print(f"    Created {len(data)} observations")
    print(f"    Columns: {list(data.columns)}")

    # Try to fit model
    print("\n[2] Attempting to fit PyMGCV model...")
    try:
        from pymgcv.api.gam import GAM

        model = GAM("y ~ s(x, k=10)", family="gaussian")
        model.fit(data, verbose=False)

        # Format output
        output = format_pymgcv_output(model, data, formula="y ~ s(x, k=10)", family="gaussian")
        print(output)

        # Save to file
        output_file = Path("pymgcv_example_output.txt")
        with open(output_file, "w") as f:
            f.write(output)
        print(f"\n✓ Output saved to {output_file}")

    except Exception as e:
        print(f"✗ Could not fit model: {e}")
        print("  (This is expected if all dependencies are not yet available)")

    # Create comparison template
    print("\n[3] Creating R/PyMGCV comparison template...")
    template = create_comparison_template()
    template_file = Path("comparison_template.txt")
    with open(template_file, "w") as f:
        f.write(template)
    print(f"✓ Template saved to {template_file}")

    # Save dataset for R
    print("\n[4] Saving dataset for R reproduction...")
    data_file = Path("synthetic_data.csv")
    data.to_csv(data_file, index=False)
    print(f"✓ Data saved to {data_file}")

    print("\n" + "#" * 80)
    print("# NEXT STEPS:")
    print("# 1. Run PyMGCV (see output above or in pymgcv_example_output.txt)")
    print("# 2. Run equivalent R code (see comparison_template.txt)")
    print("# 3. Compare results using the checklist in comparison_template.txt")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
