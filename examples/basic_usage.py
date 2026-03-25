"""
examples/basic_usage.py — Minimal pymgcv usage examples.

Run:
    python examples/basic_usage.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pymgcv import GAM


def gaussian_gam() -> None:
    """Fit a simple Gaussian GAM with one smooth term."""
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + np.random.normal(0, 0.3, n)
    df = pd.DataFrame({"x": x, "y": y})

    model = GAM("y ~ s(x)", data=df)
    model.fit()
    print("=== Gaussian GAM ===")
    print(model.summary())

    preds = model.predict(df, scale="response")
    print(f"Predictions (first 5): {preds[:5]}\n")


def poisson_gam() -> None:
    """Fit a Poisson GAM for count data."""
    np.random.seed(123)
    n = 300
    x = np.random.uniform(0, 5, n)
    mu = np.exp(0.5 + 0.3 * np.sin(x * 2))
    y = np.random.poisson(mu)
    df = pd.DataFrame({"x": x, "y": y})

    model = GAM("y ~ s(x)", family="poisson", data=df)
    model.fit()
    print("=== Poisson GAM ===")
    print(model.summary())


def multi_smooth_gam() -> None:
    """Fit a GAM with multiple smooth terms."""
    np.random.seed(7)
    n = 400
    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2**2 + np.random.normal(0, 0.2, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    model = GAM("y ~ s(x1) + s(x2, bs='cr')", data=df)
    model.fit()
    print("=== Multi-smooth GAM ===")
    print(model.summary())


if __name__ == "__main__":
    gaussian_gam()
    poisson_gam()
    multi_smooth_gam()
