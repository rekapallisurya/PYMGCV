"""Integration tests for all distribution families with GAM fitting (Phase 2A)."""

import numpy as np
import pytest

from pymgcv.distributions import (
    BinomialFamily,
    GammaFamily,
    GaussianFamily,
    InverseGaussianFamily,
    NegativeBinomialFamily,
    PoissonFamily,
    TweedieFamily,
)
from pymgcv.optimizer.pirls import PIRLSSolver

np.random.seed(123)
_n = 50
_X = np.column_stack([np.ones(_n), np.random.randn(_n)])


@pytest.mark.parametrize(
    "family, y, offset, weights",
    [
        (GaussianFamily(), np.random.randn(_n), None, None),
        (PoissonFamily(), np.random.poisson(2.0, _n).astype(float), None, None),
        (BinomialFamily(), np.random.binomial(1, 0.5, _n).astype(float), None, None),
        (GammaFamily(), np.random.gamma(2.0, 2.0, _n), None, None),
        (TweedieFamily(), np.random.gamma(2.0, 2.0, _n), None, None),
        (
            NegativeBinomialFamily(),
            np.random.negative_binomial(2, 0.5, _n).astype(float),
            None,
            None,
        ),
        (InverseGaussianFamily(), np.random.gamma(2.0, 2.0, _n), None, None),
    ],
)
def test_gam_family_fit(family, y, offset, weights):
    """Fit via PIRLS for each family; check convergence and coefficient shape."""
    X = _X.copy()
    S_list = [np.zeros((X.shape[1], X.shape[1]))]  # No penalty

    solver = PIRLSSolver(X, y, family, S_list, offset=offset, weights=weights)
    beta = solver.solve(max_iter=50, verbose=False)

    # Coefficients must have correct shape and be finite
    assert beta.shape[0] == X.shape[1]
    assert np.all(np.isfinite(beta))

    # Predict: linear predictor → response scale
    off = offset if offset is not None else np.zeros(X.shape[0])
    eta = X @ beta + off
    mu = family.linkinv(eta)
    assert mu.shape == (X.shape[0],)
    assert np.all(np.isfinite(mu))
