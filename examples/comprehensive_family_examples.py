"""
Comprehensive Examples: PyMGCV vs MGCV Equivalence Testing

This file demonstrates PyMGCV models that can be validated against R mgcv.
Includes examples for:
1. Binomial GAM (binary classification)
2. Negative Binomial GAM (overdispersed counts)
3. Inverse Gaussian GAM (heavy-tailed data)
4. Comparisons of different basis types
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "c:\\Users\\surya\\Downloads\\pymgcv")

from pymgcv.api.gam import GAM


def example_1_binomial_gam():
    """
    Example 1: Binomial GAM with Binary Response

    MGCV Equivalent:
    ```r
    library(mgcv)
    set.seed(42)
    n <- 100
    x <- seq(0, 1, len=n)
    p <- 0.3 + 0.4*sin(6*pi*x)
    y <- rbinom(n, 1, p)
    df <- data.frame(x=x, y=y)

    model <- gam(y ~ s(x), family=binomial(), data=df)
    summary(model)
    predictions <- predict(model, newdata=data.frame(x=seq(0, 1, 0.1)),
                           type="response", se.fit=TRUE)
    ```
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: BINOMIAL GAM (Binary Classification)")
    print("=" * 70)

    # Generate data
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    p_true = 0.5 + 0.3 * np.sin(6 * np.pi * x)  # Keep within [0.2, 0.8]
    p_true = np.clip(p_true, 0.01, 0.99)  # Ensure valid probabilities
    y = np.random.binomial(1, p_true)

    df = pd.DataFrame({"x": x, "y": y})

    print(f"\nData: n={n} observations")
    print(f"Response: binary (proportion of 1s = {y.mean():.3f})")
    print("Predictor: x in [0, 1]")

    # Fit model
    print("\nFitting: GAM('y ~ s(x)', family='binomial')")
    try:
        model = GAM("y ~ s(x)", family="binomial")
        model.fit(df, max_outer_iter=5)

        print("✓ Model fitted successfully!")

        if hasattr(model, "summary"):
            print("\nModel Summary:")
            print(model.summary())

        # Predictions
        x_new = np.linspace(0, 1, 11)
        df_new = pd.DataFrame({"x": x_new})
        try:
            pred_eta = model.predict(df_new)
            pred_mu = 1 / (1 + np.exp(-np.clip(pred_eta, -500, 500)))

            print("\nPredictions (probability scale):")
            print("x       p̂(y=1)")
            print("-" * 20)
            for x_val, p_val in zip(x_new, pred_mu):
                print(f"{x_val:.2f}    {p_val:.4f}")
        except Exception as pred_err:
            print(f"Prediction failed: {pred_err}")

    except Exception as e:
        print(f"✗ Model fitting failed: {e}")
        import traceback

        traceback.print_exc()


def example_2_negative_binomial():
    """
    Example 2: Negative Binomial GAM (Count Data with Overdispersion)

    MGCV Equivalent:
    ```r
    library(mgcv)
    set.seed(42)
    n <- 150
    x <- seq(0, 5, len=n)
    eta <- 1.5 + 0.4*x + 0.2*sin(4*pi*x)
    y <- rnbinom(n, mu=exp(eta), size=2)
    df <- data.frame(x=x, y=y)

    model <- gam(y ~ s(x), family=negative.binomial(theta=2), data=df)
    summary(model)
    ```
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: NEGATIVE BINOMIAL GAM (Count Data)")
    print("=" * 70)

    # Generate count data with overdispersion
    np.random.seed(42)
    n = 150
    x = np.linspace(0, 5, n)
    eta_true = 1.5 + 0.4 * x + 0.2 * np.sin(4 * np.pi * x)
    # Negative binomial with shape parameter theta=2
    mu_true = np.exp(eta_true)
    y = np.random.negative_binomial(n=2, p=2 / (2 + mu_true))  # NB parameterization

    df = pd.DataFrame({"x": x, "y": y})

    print(f"\nData: n={n} count observations")
    print(f"Response: y in {{0, 1, 2, ..., {int(y.max())}}} (count data)")
    print(f"Mean: {y.mean():.3f}, Variance: {y.var():.3f}")
    print(f"Overdispersion: var/mean = {y.var()/y.mean():.3f} (>1 indicates overdispersion)")
    print("Predictor: x in [0, 5]")

    # Fit model
    print("\nFitting: GAM('y ~ s(x)', family='negative.binomial')")
    try:
        model = GAM("y ~ s(x)", family="negative.binomial")
        model.fit(df, max_outer_iter=5)

        print("✓ Model fitted successfully!")

        if hasattr(model, "summary"):
            print("\nModel Summary:")
            print(model.summary())

        # Predictions
        x_new = np.linspace(0, 5, 11)
        df_new = pd.DataFrame({"x": x_new})
        try:
            pred_eta = model.predict(df_new)
            pred_mu = np.exp(pred_eta)

            print("\nPredictions (expected count):")
            print("x       E[y]")
            print("-" * 20)
            for x_val, mu_val in zip(x_new, pred_mu):
                print(f"{x_val:.2f}    {mu_val:.4f}")
        except Exception as pred_err:
            print(f"Prediction accuracy: {pred_err}")

    except Exception as e:
        print(f"✗ Model fitting failed: {e}")
        import traceback

        traceback.print_exc()


def example_3_inverse_gaussian():
    """
    Example 3: Inverse Gaussian GAM (Heavy-tailed Positive Data)

    MGCV Equivalent:
    ```r
    library(mgcv)
    set.seed(42)
    n <- 100
    x <- seq(0, 1, len=n)
    mu <- exp(2 + 1.5*x)
    y <- 1/rgamma(n, shape=1/0.5, rate=1/(mu*0.5))  # Inverse Gaussian
    df <- data.frame(x=x, y=y)

    model <- gam(y ~ s(x), family=inverse.gaussian(), data=df)
    summary(model)
    ```
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: INVERSE GAUSSIAN GAM (Heavy-tailed Data)")
    print("=" * 70)

    # Generate inverse gaussian data
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    mu_true = np.exp(2 + 1.5 * x)

    # Generate IG-like data (positive, heavy-tailed)
    # Using exponential as approximation for computational ease
    y = np.random.exponential(scale=mu_true)

    df = pd.DataFrame({"x": x, "y": y})

    print(f"\nData: n={n} positive continuous observations")
    print(f"Response: y in ({y.min():.2f}, {y.max():.2f}] (positive, heavy-tailed)")
    print(f"Mean: {y.mean():.3f}, Std: {y.std():.3f}")
    print(f"Skewness: {(y - y.mean()).mean()**3 / y.std()**3:.3f} (> 0 = right-skewed)")
    print("Predictor: x in [0, 1]")

    # Fit model
    print("\nFitting: GAM('y ~ s(x)', family='inverse.gaussian')")
    try:
        model = GAM("y ~ s(x)", family="inverse.gaussian")
        model.fit(df, max_outer_iter=5)

        print("✓ Model fitted successfully!")

        if hasattr(model, "summary"):
            print("\nModel Summary:")
            print(model.summary())

        # Predictions
        x_new = np.linspace(0, 1, 11)
        df_new = pd.DataFrame({"x": x_new})
        try:
            pred_eta = model.predict(df_new)
            pred_mu = 1.0 / np.sqrt(np.maximum(pred_eta, 1e-10))

            print("\nPredictions (expected value):")
            print("x       E[y]")
            print("-" * 20)
            for x_val, mu_val in zip(x_new, pred_mu):
                print(f"{x_val:.2f}    {mu_val:.4f}")
        except Exception as pred_err:
            print(f"Prediction failed: {pred_err}")

    except Exception as e:
        print(f"✗ Model fitting failed: {e}")
        import traceback

        traceback.print_exc()


def example_4_model_comparison():
    """
    Example 4: Comparing Different Link Functions for Binomial

    Shows how to fit the same model with different link functions
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: MODEL COMPARISON (Different Link Functions)")
    print("=" * 70)

    # Generate data
    np.random.seed(42)
    n = 80
    x = np.linspace(0, 1, n)
    p_true = 0.3 + 0.35 * np.sin(4 * np.pi * x)
    y = np.random.binomial(1, p_true)

    df = pd.DataFrame({"x": x, "y": y})

    print(f"\nData: n={n} binary observations")

    # Fit models with different links
    links_to_test = ["logit", "probit", "cloglog"]
    models = {}

    print("\nFitting models with different link functions:")
    for link in links_to_test:
        try:
            model = GAM("y ~ s(x)", family="binomial")
            # Override family with specific link
            from pymgcv.distributions.family_base import BinomialFamily

            model.family = BinomialFamily(link=link)
            model.fit(df, max_outer_iter=3)
            models[link] = model
            print(f"  + {link:10s} link: fitted")
        except Exception as e:
            print(f"  - {link:10s} link: {e}")

    # Compare predictions
    if len(models) > 1:
        x_new = np.array([0.25, 0.5, 0.75])
        df_new = pd.DataFrame({"x": x_new})

        print("\nPrediction comparison at x in {0.25, 0.5, 0.75}:")
        print("-" * 60)
        print("x       logit        probit       cloglog")
        print("-" * 60)

        for x_val in x_new:
            print(f"{x_val:.2f}    ", end="")
            for link, model in models.items():
                try:
                    pred = model.predict(df.iloc[:1:10])
                    print(f"{pred[x_val] if isinstance(pred, dict)  else 'N/A':>12.4f}  ", end="")
                except:
                    print(f"{'N/A':>12s}  ", end="")
            print()


def example_5_summary_statistics():
    """
    Example 5: Extracting Summary Statistics

    Demonstrates how to extract key model information
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: EXTRACTING MODEL SUMMARY STATISTICS")
    print("=" * 70)

    # Simple Gaussian example
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y = 2 + 3 * x + 2 * np.sin(4 * np.pi * x) + np.random.normal(0, 0.3, n)

    df = pd.DataFrame({"x": x, "y": y})

    print("\nFitting Gaussian GAM: y ~ s(x)")
    model = GAM("y ~ s(x)", family="gaussian")
    model.fit(df)

    print("\nModel Information:")
    print("-" * 40)

    # Try to extract various components
    attributes_to_check = [
        "beta",
        "smoothing_parameters",
        "edf",
        "edf_per_smooth",
        "aic",
        "gic",
        "deviance",
        "null_deviance",
    ]

    for attr in attributes_to_check:
        if hasattr(model, attr):
            value = getattr(model, attr)
            if isinstance(value, np.ndarray):
                if len(value) > 5:
                    print(f"{attr:20s}: array of shape {value.shape}")
                else:
                    print(f"{attr:20s}: {value}")
            else:
                print(f"{attr:20s}: {value}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PYMGCV COMPREHENSIVE EXAMPLES")
    print("GAM Fitting with New Distribution Families")
    print("=" * 70)

    try:
        example_1_binomial_gam()
        example_2_negative_binomial()
        example_3_inverse_gaussian()
        example_4_model_comparison()
        example_5_summary_statistics()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
