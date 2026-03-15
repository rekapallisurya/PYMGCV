"""Tests for diagnostics modules: residuals, influence, concurvity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymgcv.api.gam import GAM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=120, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture(scope='module')
def fitted_gam():
    df = _make_data()
    m = GAM('y ~ s(x)', data=df, family='gaussian')
    m.fit()
    return m, df


# ---------------------------------------------------------------------------
# compute_residuals tests
# ---------------------------------------------------------------------------

class TestComputeResiduals:

    def test_response_residuals(self, fitted_gam):
        from pymgcv.diagnostics.residuals import compute_residuals
        m, df = fitted_gam
        mu = m.predict(df, scale='response')
        y = df['y'].values
        r = compute_residuals(y, mu, m.family, type='response')
        np.testing.assert_allclose(r, y - mu)

    def test_pearson_residuals_shape(self, fitted_gam):
        from pymgcv.diagnostics.residuals import compute_residuals
        m, df = fitted_gam
        mu = m.predict(df, scale='response')
        y = df['y'].values
        r = compute_residuals(y, mu, m.family, type='pearson')
        assert r.shape == y.shape

    def test_deviance_residuals_shape(self, fitted_gam):
        from pymgcv.diagnostics.residuals import compute_residuals
        m, df = fitted_gam
        mu = m.predict(df, scale='response')
        y = df['y'].values
        r = compute_residuals(y, mu, m.family, type='deviance')
        assert r.shape == y.shape

    def test_standardized_residuals_shape(self, fitted_gam):
        from pymgcv.diagnostics.residuals import compute_residuals
        m, df = fitted_gam
        mu = m.predict(df, scale='response')
        y = df['y'].values
        r = compute_residuals(y, mu, m.family, type='standardized')
        assert r.shape == y.shape

    def test_unknown_type_raises(self, fitted_gam):
        from pymgcv.diagnostics.residuals import compute_residuals
        m, df = fitted_gam
        mu = m.predict(df, scale='response')
        y = df['y'].values
        with pytest.raises(ValueError, match='Unknown residual type'):
            compute_residuals(y, mu, m.family, type='bad_type')

    def test_no_nans(self, fitted_gam):
        from pymgcv.diagnostics.residuals import compute_residuals
        m, df = fitted_gam
        mu = m.predict(df, scale='response')
        y = df['y'].values
        for t in ('response', 'pearson', 'deviance', 'standardized'):
            r = compute_residuals(y, mu, m.family, type=t)
            assert not np.any(np.isnan(r)), f'NaN in {t} residuals'

    def test_response_residuals_no_family(self):
        from pymgcv.diagnostics.residuals import compute_residuals
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.1, 1.9, 3.2])
        r = compute_residuals(y, mu, family=None, type='response')
        np.testing.assert_allclose(r, y - mu)

    def test_deviance_sign(self, fitted_gam):
        """Deviance residuals should be positive when y > mu."""
        from pymgcv.diagnostics.residuals import compute_residuals
        m, df = fitted_gam
        mu = m.predict(df, scale='response')
        y = df['y'].values
        r = compute_residuals(y, mu, m.family, type='deviance')
        positive = y > mu
        assert np.all(r[positive] >= 0), 'deviance residuals should be +ve when y>mu'
        assert np.all(r[~positive] <= 0), 'deviance residuals should be -ve when y<mu'


# ---------------------------------------------------------------------------
# ResidualDiagnostics class tests
# ---------------------------------------------------------------------------

class TestResidualDiagnostics:

    def test_instantiation(self, fitted_gam):
        from pymgcv.diagnostics.residuals import ResidualDiagnostics
        m, _ = fitted_gam
        rd = ResidualDiagnostics(m)
        assert rd.model is m

    def test_has_all_residual_types(self, fitted_gam):
        from pymgcv.diagnostics.residuals import ResidualDiagnostics
        m, _ = fitted_gam
        rd = ResidualDiagnostics(m)
        for t in ('response', 'pearson', 'deviance', 'standardized'):
            assert t in rd.residuals

    def test_summary_is_string(self, fitted_gam):
        from pymgcv.diagnostics.residuals import ResidualDiagnostics
        m, _ = fitted_gam
        rd = ResidualDiagnostics(m)
        s = rd.summary()
        assert isinstance(s, str)
        assert 'Residual' in s

    def test_get_residuals(self, fitted_gam):
        from pymgcv.diagnostics.residuals import ResidualDiagnostics
        m, _ = fitted_gam
        rd = ResidualDiagnostics(m)
        r = rd.get_residuals('pearson')
        assert len(r) == len(m.data)

    def test_qq_plot_data(self, fitted_gam):
        from pymgcv.diagnostics.residuals import ResidualDiagnostics
        m, _ = fitted_gam
        rd = ResidualDiagnostics(m)
        theoretical, sample = rd.qq_plot_data('deviance')
        assert len(theoretical) == len(sample) == len(m.data)
        assert np.all(np.diff(sample) >= 0), 'sample quantiles should be sorted'

    def test_unfitted_model_raises(self):
        from pymgcv.diagnostics.residuals import ResidualDiagnostics
        m = GAM('y ~ s(x)', data=_make_data(), family='gaussian')
        with pytest.raises(RuntimeError, match='not fitted|not fit'):
            ResidualDiagnostics(m)


# ---------------------------------------------------------------------------
# influence.py tests
# ---------------------------------------------------------------------------

class TestInfluence:

    def test_leverage_from_diagonal(self):
        from pymgcv.diagnostics.influence import leverage
        H = np.eye(5) * 0.3
        lev = leverage(H)
        np.testing.assert_allclose(lev, np.full(5, 0.3))

    def test_cooks_distance_shape(self):
        from pymgcv.diagnostics.influence import cooks_distance
        n = 30
        resids = np.random.default_rng(0).normal(size=n)
        lev = np.full(n, 0.1)
        cd = cooks_distance(resids, lev, scale=1.0, p=5)
        assert cd.shape == (n,)

    def test_cooks_distance_non_negative(self):
        from pymgcv.diagnostics.influence import cooks_distance
        rng = np.random.default_rng(1)
        n = 50
        resids = rng.normal(size=n)
        lev = rng.uniform(0.01, 0.3, n)
        cd = cooks_distance(resids, lev, scale=1.0, p=4)
        assert np.all(cd >= 0)

    def test_dfbetas_shape(self):
        from pymgcv.diagnostics.influence import dfbetas
        rng = np.random.default_rng(2)
        n, p = 40, 5
        X = rng.normal(size=(n, p))
        r = rng.normal(size=n)
        lev = np.full(n, 1.0 / n)
        D = dfbetas(X, r, lev)
        assert D.shape == (n, p)

    def test_high_residual_high_cooks(self):
        """Observations with large residuals get higher Cook's distance."""
        from pymgcv.diagnostics.influence import cooks_distance
        n = 20
        resids = np.ones(n) * 0.1
        resids[0] = 10.0  # outlier
        lev = np.full(n, 0.1)
        cd = cooks_distance(resids, lev)
        assert cd[0] > cd[1], 'outlier should have highest Cook\'s distance'


# ---------------------------------------------------------------------------
# concurvity tests
# ---------------------------------------------------------------------------

class TestConcurvity:

    def _make_smooth_data(self):
        rng = np.random.default_rng(5)
        n = 100
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        return x1[:, None], x2[:, None]

    def test_returns_dict_with_keys(self):
        from pymgcv.diagnostics.concurvity import concurvity
        S1, S2 = self._make_smooth_data()
        X = np.hstack([S1, S2])
        indices = [slice(0, 1), slice(1, 2)]
        result = concurvity(X, indices)
        assert 'concurvity_matrix' in result
        assert 'overall' in result
        assert 'pairwise' in result

    def test_matrix_shape(self):
        from pymgcv.diagnostics.concurvity import concurvity
        S1, S2 = self._make_smooth_data()
        X = np.hstack([S1, S2])
        indices = [slice(0, 1), slice(1, 2)]
        result = concurvity(X, indices)
        assert result['concurvity_matrix'].shape == (2, 2)

    def test_diagonal_is_one(self):
        from pymgcv.diagnostics.concurvity import concurvity
        S1, S2 = self._make_smooth_data()
        X = np.hstack([S1, S2])
        indices = [slice(0, 1), slice(1, 2)]
        result = concurvity(X, indices)
        np.testing.assert_allclose(np.diag(result['concurvity_matrix']), 1.0)

    def test_overall_in_range(self):
        from pymgcv.diagnostics.concurvity import concurvity
        S1, S2 = self._make_smooth_data()
        X = np.hstack([S1, S2])
        indices = [slice(0, 1), slice(1, 2)]
        result = concurvity(X, indices)
        assert 0.0 <= result['overall'] <= 1.0

    def test_identical_smooths_high_concurvity(self):
        """Two identical columns should show high concurvity."""
        from pymgcv.diagnostics.concurvity import concurvity
        rng = np.random.default_rng(9)
        col = rng.uniform(size=(80, 3))
        X = np.hstack([col, col])  # duplicate
        indices = [slice(0, 3), slice(3, 6)]
        result = concurvity(X, indices)
        # Off-diagonal element should be near 1
        assert result['concurvity_matrix'][0, 1] > 0.9

    def test_orthogonal_smooths_low_concurvity(self):
        """Nearly orthogonal smooths should have low concurvity."""
        from pymgcv.diagnostics.concurvity import concurvity
        rng = np.random.default_rng(11)
        n = 200
        # Create two clearly orthogonal bases: sine and cosine at different frequencies
        t = np.linspace(0, 2 * np.pi, n)
        col1 = np.column_stack([np.sin(t), np.sin(2 * t)])
        col2 = np.column_stack([np.cos(3 * t), np.cos(5 * t)])
        X = np.hstack([col1, col2])
        indices = [slice(0, 2), slice(2, 4)]
        result = concurvity(X, indices)
        # Should be well below 0.8 (sin/cos at different frequencies are near-orthogonal)
        assert result['concurvity_matrix'][0, 1] < 0.8
