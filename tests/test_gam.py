"""
tests/test_gam.py — Core GAM fitting and prediction tests.

Covers:
    - Gaussian GAM fit + predict
    - Poisson GAM fit + predict
    - Multiple smooth terms
    - Summary output
    - Coefficient access
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymgcv import GAM


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture()
def gaussian_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + np.random.normal(0, 0.3, n)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture()
def poisson_data() -> pd.DataFrame:
    np.random.seed(123)
    n = 300
    x = np.random.uniform(0, 5, n)
    mu = np.exp(0.5 + 0.3 * np.sin(x * 2))
    y = np.random.poisson(mu)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture()
def multi_data() -> pd.DataFrame:
    np.random.seed(7)
    n = 400
    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2**2 + np.random.normal(0, 0.2, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


# ── Gaussian GAM ────────────────────────────────────────────────────

class TestGaussianGAM:
    def test_fit_succeeds(self, gaussian_data: pd.DataFrame) -> None:
        model = GAM("y ~ s(x)", data=gaussian_data)
        model.fit()
        assert hasattr(model, "beta") and model.beta is not None

    def test_predict_shape(self, gaussian_data: pd.DataFrame) -> None:
        model = GAM("y ~ s(x)", data=gaussian_data)
        model.fit()
        preds = model.predict(gaussian_data, scale="response")
        assert len(preds) == len(gaussian_data)

    def test_predict_reasonable(self, gaussian_data: pd.DataFrame) -> None:
        """Predictions should be in a sensible range for sin(x) data."""
        model = GAM("y ~ s(x)", data=gaussian_data)
        model.fit()
        preds = model.predict(gaussian_data, scale="response")
        assert np.all(np.isfinite(preds))
        assert np.abs(np.mean(preds)) < 2.0  # centred near 0

    def test_summary_returns_string(self, gaussian_data: pd.DataFrame) -> None:
        model = GAM("y ~ s(x)", data=gaussian_data)
        model.fit()
        s = model.summary()
        assert isinstance(s, str)
        assert "s(x)" in s


# ── Poisson GAM ─────────────────────────────────────────────────────

class TestPoissonGAM:
    def test_fit_succeeds(self, poisson_data: pd.DataFrame) -> None:
        model = GAM("y ~ s(x)", family="poisson", data=poisson_data)
        model.fit()

    def test_predict_positive(self, poisson_data: pd.DataFrame) -> None:
        model = GAM("y ~ s(x)", family="poisson", data=poisson_data)
        model.fit()
        preds = model.predict(poisson_data, scale="response")
        assert np.all(preds >= 0)


# ── Multi-smooth GAM ────────────────────────────────────────────────

class TestMultiSmoothGAM:
    def test_fit_two_smooths(self, multi_data: pd.DataFrame) -> None:
        model = GAM("y ~ s(x1) + s(x2)", data=multi_data)
        model.fit()
        s = model.summary()
        assert "s(x1)" in s
        assert "s(x2)" in s

    def test_different_basis(self, multi_data: pd.DataFrame) -> None:
        model = GAM("y ~ s(x1, bs='cr') + s(x2, bs='tp')", data=multi_data)
        model.fit()
        preds = model.predict(multi_data, scale="response")
        assert len(preds) == len(multi_data)
