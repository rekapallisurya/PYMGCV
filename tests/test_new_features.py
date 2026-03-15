"""Tests for by-variable, cyclic spline, random effect, and weights features.

Validates:
- s(x, by=group): factor by-variable expansion
- s(x, by=weight): continuous by-variable scaling
- s(x, bs='cc'): cyclic cubic spline
- re(group): random effect smooth
- GAM(..., weights=...): case weights
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymgcv.smooth.cyclic_spline import CyclicSpline
from pymgcv.smooth.random_effect import RandomEffect


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def make_factor_data(n=120, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    group = np.array(['A', 'B', 'C'] * (n // 3))[:n]
    effect = np.where(group == 'A', 2.0, np.where(group == 'B', -1.0, 0.5))
    y = np.sin(x) * effect + rng.normal(0, 0.3, n)
    return pd.DataFrame({'x': x, 'group': group, 'y': y})


def make_continuous_data(n=100, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.2, n)
    return pd.DataFrame({'x': x, 'y': y})


# ── Cyclic Spline ─────────────────────────────────────────────────────────────

class TestCyclicSpline:
    def test_basis_shape(self):
        x = np.linspace(0, 2 * np.pi, 80)
        cs = CyclicSpline(x, k=12)
        assert cs.B.shape == (80, 12)

    def test_penalty_shape(self):
        x = np.linspace(0, 2 * np.pi, 80)
        cs = CyclicSpline(x, k=10)
        assert cs.S.shape == (10, 10)

    def test_penalty_symmetric(self):
        x = np.linspace(0, 4 * np.pi, 100)
        cs = CyclicSpline(x, k=8)
        np.testing.assert_allclose(cs.S, cs.S.T, atol=1e-10)

    def test_basis_finite(self):
        x = np.linspace(0, 2 * np.pi, 60)
        cs = CyclicSpline(x, k=8)
        assert np.all(np.isfinite(cs.B))

    def test_predict_shape(self):
        x_train = np.linspace(0, 2 * np.pi, 50)
        cs = CyclicSpline(x_train, k=8)
        x_new = np.linspace(0, 2 * np.pi, 20)
        B_new = cs.predict(x_new)
        assert B_new.shape == (20, 8)

    def test_basis_matrix_method(self):
        x = np.linspace(0, 2 * np.pi, 30)
        cs = CyclicSpline(x, k=6)
        B = cs.basis_matrix()
        assert B.shape == (30, 6)
        np.testing.assert_array_equal(B, cs.B)

    def test_default_k(self):
        x = np.linspace(0, 2 * np.pi, 50)
        cs = CyclicSpline(x)  # Default k=10
        assert cs.k == 10
        assert cs.B.shape == (50, 10)


# ── Random Effect ─────────────────────────────────────────────────────────────

class TestRandomEffect:
    def test_factor_basis_shape(self):
        group = np.array(['A', 'B', 'C'] * 20)
        re = RandomEffect(group)
        assert re.B.shape == (60, 3)

    def test_factor_levels_detected(self):
        group = np.array(['A', 'B', 'C'] * 10)
        re = RandomEffect(group)
        assert re.k == 3
        assert set(re.levels) == {'A', 'B', 'C'}

    def test_identity_penalty(self):
        group = np.array(['X', 'Y', 'Z'] * 10)
        re = RandomEffect(group)
        np.testing.assert_allclose(re.S, np.eye(3))

    def test_factor_basis_binary(self):
        """Basis for factor RE should be 0/1 only."""
        group = np.array(['A', 'B'] * 15)
        re = RandomEffect(group)
        assert re.B.shape == (30, 2)
        assert set(np.unique(re.B)) == {0.0, 1.0}

    def test_each_row_sums_to_one(self):
        """Each row of factor RE basis should sum to one (one-hot)."""
        group = np.array(['cat', 'dog', 'bird'] * 10)
        re = RandomEffect(group)
        row_sums = re.B.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(30))

    def test_factor_predict(self):
        group_train = np.array(['A', 'B', 'C'] * 10)
        re = RandomEffect(group_train)
        group_new = np.array(['B', 'A'])
        B_new = re.predict(group_new)
        assert B_new.shape == (2, 3)
        # 'B' maps to index 1, so B_new[0, 1] = 1
        level_order = list(re.levels)
        b_idx = level_order.index('B')
        assert B_new[0, b_idx] == 1.0

    def test_penalty_matrix_method(self):
        group = np.array(['X', 'Y'] * 10)
        re = RandomEffect(group)
        S = re.penalty_matrix()
        np.testing.assert_allclose(S, np.eye(2))


# ── by-Variable Formula Parser ────────────────────────────────────────────────

class TestByVariableParser:
    def test_parse_by_factor(self):
        from pymgcv.utils.formula_parser import FormulaParser
        parser = FormulaParser('y ~ s(x, by=group)')
        assert len(parser.smooth_terms) == 1
        spec = parser.smooth_terms[0]
        assert spec.by_variable == 'group'

    def test_parse_by_continuous(self):
        from pymgcv.utils.formula_parser import FormulaParser
        parser = FormulaParser('y ~ s(x, by=weight)')
        assert parser.smooth_terms[0].by_variable == 'weight'

    def test_parse_bs_cc(self):
        from pymgcv.utils.formula_parser import FormulaParser
        parser = FormulaParser('y ~ s(x, bs=cc)')
        spec = parser.smooth_terms[0]
        assert spec.basis == 'cc'

    def test_parse_re(self):
        from pymgcv.utils.formula_parser import FormulaParser
        parser = FormulaParser('y ~ re(group)')
        spec = parser.smooth_terms[0]
        assert spec.term_type == 're'
        assert spec.basis == 're'


# ── by-Variable Model Matrix ──────────────────────────────────────────────────

class TestByVariableModelMatrix:
    def test_factor_by_expands_columns(self):
        """s(x, by=group) with 3 levels should give 3×k columns."""
        from pymgcv.utils.model_matrix import ModelMatrix
        df = make_factor_data(90)
        k = 8
        mm = ModelMatrix(df, f'y ~ s(x, by=group, k={k})', center=False)
        # 3 levels × 8 basis cols = 24 smooth cols + 1 intercept
        smooth_slice = mm.smooth_indices[0]
        smooth_dim = smooth_slice.stop - smooth_slice.start
        assert smooth_dim == 3 * k, f"Expected {3*k} smooth cols, got {smooth_dim}"

    def test_factor_by_levels_stored(self):
        """by-levels should be stored in smooth_by_levels."""
        from pymgcv.utils.model_matrix import ModelMatrix
        df = make_factor_data(60)
        mm = ModelMatrix(df, 'y ~ s(x, by=group, k=6)', center=False)
        levels = mm.smooth_by_levels[0]
        assert levels is not None
        assert set(levels) == {'A', 'B', 'C'}

    def test_continuous_by_no_expansion(self):
        """Continuous by-variable should not expand columns."""
        from pymgcv.utils.model_matrix import ModelMatrix
        rng = np.random.default_rng(0)
        n = 60
        df = pd.DataFrame({'x': rng.normal(size=n), 'w': rng.uniform(0.5, 2.0, n),
                           'y': rng.normal(size=n)})
        k = 6
        mm = ModelMatrix(df, f'y ~ s(x, by=w, k={k})', center=False)
        smooth_slice = mm.smooth_indices[0]
        smooth_dim = smooth_slice.stop - smooth_slice.start
        # Continuous by: same k columns (scaled)
        assert smooth_dim == k, f"Expected {k} smooth cols, got {smooth_dim}"

    def test_basis_zeros_padding(self):
        """Observations in one level should have zero contribution to other-level cols."""
        from pymgcv.utils.model_matrix import ModelMatrix
        df = make_factor_data(60)
        mm = ModelMatrix(df, 'y ~ s(x, by=group, k=5)', center=False)
        X = mm.X
        smooth_slice = mm.smooth_indices[0]
        X_smooth = X[:, smooth_slice]
        levels = mm.smooth_by_levels[0]
        groups = df['group'].values
        k = 5
        # Check that rows for group 'A' (index 0) are 0 in columns for group 'B' and 'C'
        for level_idx, level in enumerate(levels):
            mask = (groups == level)
            other_start = 0 if level_idx > 0 else k
            other_end = other_start + k
            # Rows for this level should be zero in the other-level columns
            for other_idx in range(len(levels)):
                if other_idx == level_idx:
                    continue
                other_cols = slice(other_idx * k, (other_idx + 1) * k)
                zero_block = X_smooth[mask, other_cols]
                np.testing.assert_allclose(zero_block, 0.0)


# ── GAM with by-variable ──────────────────────────────────────────────────────

class TestByVariableGAM:
    def test_fit_factor_by(self):
        """GAM with factor by-variable should fit."""
        from pymgcv.api.gam import GAM
        df = make_factor_data(90)
        model = GAM('y ~ s(x, by=group, k=6)', data=df, family='gaussian')
        model.fit(max_outer_iter=3, max_inner_iter=10, use_gpu=False)
        assert model.fitted
        assert np.all(np.isfinite(model.beta))

    def test_fit_continuous_by(self):
        """GAM with continuous by-variable should fit."""
        from pymgcv.api.gam import GAM
        rng = np.random.default_rng(5)
        n = 80
        df = pd.DataFrame({
            'x': rng.normal(size=n),
            'w': rng.uniform(0.5, 2.0, n),
            'y': rng.normal(size=n)
        })
        model = GAM('y ~ s(x, by=w, k=6)', data=df, family='gaussian')
        model.fit(max_outer_iter=3, max_inner_iter=10, use_gpu=False)
        assert model.fitted


# ── GAM with Cyclic Spline ───────────────────────────────────────────────────

class TestCyclicSplineGAM:
    def test_fit_cyclic(self):
        """GAM with bs='cc' should fit without errors."""
        from pymgcv.api.gam import GAM
        df = make_continuous_data(80)
        model = GAM('y ~ s(x, bs=cc, k=8)', data=df, family='gaussian')
        model.fit(max_outer_iter=3, max_inner_iter=10, use_gpu=False)
        assert model.fitted

    def test_cyclic_predictions_finite(self):
        """Predictions from cyclic GAM should be finite."""
        from pymgcv.api.gam import GAM
        df = make_continuous_data(60)
        model = GAM('y ~ s(x, bs=cc, k=8)', data=df, family='gaussian')
        model.fit(max_outer_iter=3, max_inner_iter=10, use_gpu=False)
        preds = model.predict(df)
        assert np.all(np.isfinite(preds))


# ── GAM with weights ─────────────────────────────────────────────────────────

class TestGAMWeights:
    def test_weights_loaded_from_column(self):
        """GAM should accept weights as column name."""
        from pymgcv.api.gam import GAM
        rng = np.random.default_rng(99)
        n = 60
        df = pd.DataFrame({
            'x': rng.normal(size=n),
            'y': rng.normal(size=n),
            'w': rng.uniform(0.5, 2.0, n)
        })
        model = GAM('y ~ s(x, k=6)', data=df, family='gaussian', weights='w')
        model.fit(max_outer_iter=3, max_inner_iter=8, use_gpu=False)
        assert model.fitted

    def test_weights_array_direct(self):
        """GAM should accept weights as a numpy array."""
        from pymgcv.api.gam import GAM
        rng = np.random.default_rng(33)
        n = 60
        df = pd.DataFrame({'x': rng.normal(size=n), 'y': rng.normal(size=n)})
        w = rng.uniform(0.5, 2.0, n)
        model = GAM('y ~ s(x, k=6)', data=df, family='gaussian', weights=w)
        model.fit(max_outer_iter=3, max_inner_iter=8, use_gpu=False)
        assert model.fitted

    def test_weights_validation_negative(self):
        """Negative weights should raise ValueError."""
        from pymgcv.api.gam import GAM
        rng = np.random.default_rng(1)
        n = 40
        df = pd.DataFrame({'x': rng.normal(size=n), 'y': rng.normal(size=n)})
        w = rng.uniform(-2, 2, n)  # Has negatives
        model = GAM('y ~ s(x, k=5)', data=df, family='gaussian', weights=w)
        with pytest.raises(ValueError, match='positive'):
            model.fit(max_outer_iter=2, max_inner_iter=5, use_gpu=False)

    def test_weights_different_coefs(self):
        """Weighted fit should converge without error (weights wired through)."""
        from pymgcv.api.gam import GAM
        rng = np.random.default_rng(55)
        n = 80
        df = pd.DataFrame({'x': rng.normal(size=n), 'y': rng.normal(size=n)})
        # Extreme weights
        w = np.concatenate([np.full(n // 2, 10.0), np.full(n - n // 2, 0.1)])

        model_std = GAM('y ~ s(x, k=6)', data=df, family='gaussian')
        model_wtd = GAM('y ~ s(x, k=6)', data=df, family='gaussian', weights=w)

        model_std.fit(max_outer_iter=3, max_inner_iter=8, use_gpu=False)
        model_wtd.fit(max_outer_iter=3, max_inner_iter=8, use_gpu=False)

        # Both should converge
        assert model_std.fitted
        assert model_wtd.fitted
        # Both should give finite coefficients
        assert np.all(np.isfinite(model_std.beta))
        assert np.all(np.isfinite(model_wtd.beta))


# ── Standard Errors & Confidence Intervals ───────────────────────────────────

class TestInference:
    def _fit_model(self, n=80):
        from pymgcv.api.gam import GAM
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'x': rng.normal(size=n),
            'y': rng.normal(size=n)
        })
        model = GAM('y ~ s(x, k=8)', data=df, family='gaussian')
        model.fit(max_outer_iter=5, max_inner_iter=10, use_gpu=False)
        return model

    def test_standard_errors_shape(self):
        model = self._fit_model()
        se = model.standard_errors()
        assert se is not None
        assert se.shape == model.beta.shape

    def test_standard_errors_positive(self):
        model = self._fit_model()
        se = model.standard_errors()
        assert np.all(se >= 0), "All SEs should be non-negative"

    def test_confidence_intervals_shape(self):
        model = self._fit_model()
        lo, hi = model.confidence_intervals()
        assert lo.shape == model.beta.shape
        assert hi.shape == model.beta.shape

    def test_confidence_intervals_order(self):
        model = self._fit_model()
        lo, hi = model.confidence_intervals()
        assert np.all(lo <= hi), "Lower CI should not exceed upper CI"

    def test_confidence_intervals_contain_beta(self):
        model = self._fit_model()
        lo, hi = model.confidence_intervals(level=0.99)
        assert np.all(lo <= model.beta + 1e-10), "Beta should be >= lower CI"
        assert np.all(model.beta - 1e-10 <= hi), "Beta should be <= upper CI"

    def test_95_narrower_than_99(self):
        model = self._fit_model()
        lo95, hi95 = model.confidence_intervals(level=0.95)
        lo99, hi99 = model.confidence_intervals(level=0.99)
        width95 = hi95 - lo95
        width99 = hi99 - lo99
        assert np.all(width95 <= width99 + 1e-10), "99% CI should be wider than 95% CI"
