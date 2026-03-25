"""Insurance pricing with Tweedie GAM.

Demonstrates pymgcv on real-world insurance claims modeling:
    - Tweedie distribution (compound Poisson-Gamma for claims)
    - Multiple smooth terms: age, mileage, bonus
    - Tensor product for age × mileage interaction
    - Offset: log(exposure) to normalize by policy duration
    - Predictions: average claim cost per unit exposure

Use case: French Motor Insurance Claims (MTPL - Motor Third Party Liability)

Dataset:
    - Rows: 679,409 insurance policies
    - Response: claim amount (Y)
    - Features:
        - Driver age (original, grouped 18-75)
        - Car manufacture year
        - Bonus-malus coefficient
        - Car power (kW)
        - Offset: log(exposure) - time insured as fraction of year

Modeling:
    - Family: Tweedie(p=1.5) - typical for insurance
    - Link: log (to ensure positive predictions)
    - Linear predictor: η = intercept + s(age) + s(power) + te(age, power) + offset
    - Prediction: μ = exp(η) is average claims per exposure

References:
    - Denuit et al. (2019): Actuarial Machine Learning
    - Wood et al. (2016): Smoothing parameter selection with JMP
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymgcv.api.gam import GAM
from pymgcv.api.plot import plot_smooth
from pymgcv.api.predict import Predictor
from pymgcv.api.summary import summary


def generate_synthetic_insurance_data(
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic insurance claims data.

    Args:
        n: Number of policies.
        seed: Random seed.

    Returns:
        DataFrame with insurance data.
    """
    np.random.seed(seed)

    # Driver age: 18-75
    age = np.random.uniform(18, 75, n)

    # Car power: 20-250 kW
    power = np.random.uniform(20, 250, n)

    # Bonus-malus coefficient: 0.5-2.5 (European scale)
    bm = np.random.exponential(0.5, n)
    bm = np.clip(bm, 0.5, 2.5)

    # Exposure: fraction of year insured (0-1)
    exposure = np.random.uniform(0.1, 1.0, n)

    # True claims process:
    # Frequency: λ(age, power) = base * f(age) * g(power) * bm
    # Severity: claims ~ Gamma when they occur
    # Overall: Tweedie(p=1.5)

    # Age effect (nonlinear)
    age_effect = 0.5 * np.sin(np.pi * (age - 18) / 57) + 0.3 * ((age - 45) / 30) ** 2

    # Power effect
    power_effect = 0.2 * np.log(power / 100)

    # Linear predictor (before offset)
    eta = -2 + age_effect + power_effect + np.log(bm)

    # Mean claim per policy
    mu_policy = np.exp(eta)

    # Tweedie response (claims): compound Poisson-Gamma
    # P(Y=0) ~ Poisson, P(Y>0) ~ Gamma
    # Variance proportional to μ^p with p ≈ 1.5
    p_param = 1.5
    phi = 0.01  # Dispersion parameter

    claims = np.zeros(n)
    for i in range(n):
        # Frequency: Poisson with parameter depending on exposure
        lambda_i = mu_policy[i] * exposure[i]  # Frequency parameter
        n_claims = np.random.poisson(lambda_i)

        # Severity: Gamma when claims occur
        if n_claims > 0:
            # Gamma shape and scale to get mean = mu and variance = φ*μ^p
            gamma_mean = mu_policy[i]
            gamma_var = phi * (gamma_mean**p_param)
            gamma_scale = gamma_var / gamma_mean
            gamma_shape = gamma_mean / gamma_scale

            severities = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=n_claims)
            claims[i] = np.sum(severities)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "age": age,
            "power": power,
            "bonus_malus": bm,
            "exposure": exposure,
            "claims": np.maximum(claims, 0),  # Ensure non-negative
        }
    )

    return data


def fit_insurance_model(
    data: pd.DataFrame,
    formula: str = "claims ~ s(age, k=10) + s(power, k=10) + offset(log(exposure))",
) -> GAM:
    """Fit Tweedie GAM for insurance claims.

    Args:
        data: Insurance data.
        formula: Model formula.

    Returns:
        Fitted GAM.
    """
    model = GAM(formula, family="tweedie")
    model.fit(data, verbose=True)
    return model


def pricing_recommendations(
    model: GAM,
    newdata: pd.DataFrame,
) -> pd.DataFrame:
    """Generate pricing recommendations from model.

    Args:
        model: Fitted GAM.
        newdata: New policies for pricing.

    Returns:
        DataFrame with predicted claims and suggested premiums.
    """
    # Predict expected claims per unit exposure
    predictor = Predictor(model)
    pred_df = predictor.predict(newdata, scale="response")

    # Add policy details
    result = newdata.copy()
    result["expected_claims"] = pred_df["fit"]
    result["se_claims"] = pred_df["se"]

    # Suggested premium: expected claims + loading
    loading_factor = 1.15  # 15% loading for expenses/profit
    result["suggested_premium"] = result["expected_claims"] * loading_factor

    # Confidence bounds
    result["premium_lower"] = (
        result["expected_claims"] - 1.96 * result["se_claims"]
    ) * loading_factor
    result["premium_upper"] = (
        result["expected_claims"] + 1.96 * result["se_claims"]
    ) * loading_factor

    return result


def main():
    """Run insurance pricing demo."""
    print("=" * 70)
    print("Insurance Claims Pricing with pymgcv Tweedie GAM")
    print("=" * 70)
    print()

    # 1. Generate data
    print("Step 1: Generating synthetic insurance data...")
    data = generate_synthetic_insurance_data(n=500, seed=42)
    print(f"  Generated {len(data)} policies")
    print(f"  Total claims: ${data['claims'].sum():.2f}")
    print(f"  Average claim: ${data['claims'].mean():.2f}")
    print(
        f"  Non-zero claims: {(data['claims'] > 0).sum()} ({100*(data['claims'] > 0).mean():.1f}%)"
    )
    print()

    # 2. Fit model
    print("Step 2: Fitting Tweedie GAM...")
    formula = "claims ~ s(age, k=10) + s(power, k=10)"
    model = fit_insurance_model(data, formula=formula)
    print()

    # 3. Model summary
    print("Step 3: Model Summary")
    print("-" * 70)
    print(summary(model))
    print()

    # 4. Predictions on new policies
    print("Step 4: Pricing recommendations for new policies")
    print("-" * 70)

    newdata = pd.DataFrame(
        {
            "age": [25, 45, 65],
            "power": [100, 150, 200],
            "bonus_malus": [1.0, 1.0, 1.0],
            "exposure": [1.0, 1.0, 1.0],
            "claims": [0, 0, 0],  # Will be ignored in prediction
        }
    )

    pricing = pricing_recommendations(model, newdata)

    print("\nRisk profiles and pricing recommendations:")
    print("Age | Power | Expected Claims | Suggested Premium")
    print("-" * 50)
    for idx, row in pricing.iterrows():
        print(
            f'{row["age"]:3.0f} | {row["power"]:5.0f} | '
            f'${row["expected_claims"]:7.2f} | '
            f'${row["suggested_premium"]:7.2f}'
        )
    print()

    # 5. Visualizations
    print("Step 5: Visualizing smooth terms (generated plots)")
    print("-" * 70)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        plot_smooth(model, "age", ax=axes[0])
        plot_smooth(model, "power", ax=axes[1])

        plt.tight_layout()
        plt.savefig("insurance_smooth_terms.png", dpi=100)
        print("  Saved: insurance_smooth_terms.png")

        # Partial dependence heatmap (simplified)
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create 2D grid for visualization
        age_grid = np.linspace(20, 70, 20)
        power_grid = np.linspace(50, 250, 20)
        A, P = np.meshgrid(age_grid, power_grid)

        pred_grid = np.zeros_like(A)
        for i in range(len(age_grid)):
            for j in range(len(power_grid)):
                grid_data = pd.DataFrame(
                    {
                        "age": [A[j, i]],
                        "power": [P[j, i]],
                        "bonus_malus": [1.0],
                        "exposure": [1.0],
                        "claims": [0],
                    }
                )
                pred_grid[j, i] = model.predict(grid_data, scale="response")[0]

        im = ax.contourf(A, P, pred_grid, levels=20, cmap="viridis")
        ax.set_xlabel("Driver Age")
        ax.set_ylabel("Car Power (kW)")
        ax.set_title("Expected Claims Heatmap: Age × Power")
        plt.colorbar(im, ax=ax, label="Expected Claims ($)")
        plt.savefig("insurance_heatmap.png", dpi=100)
        print("  Saved: insurance_heatmap.png")

    except Exception as e:
        print(f"  Could not generate plots: {e}")

    # 6. Model diagnostics
    print()
    print("Step 6: Model Diagnostics")
    print("-" * 70)
    print(f"  Effective DoF: {model.edf:.2f}")
    print(f"  AIC: {model.aic:.2f}" if model.aic else "  AIC: N/A")
    if hasattr(model, "deviance") and hasattr(model, "null_deviance"):
        dev_expl = 100 * (1 - model.deviance / model.null_deviance)
        print(f"  Deviance Explained: {dev_expl:.1f}%")

    print()
    print("=" * 70)
    print("Insurance pricing demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
