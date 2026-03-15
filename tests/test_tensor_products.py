"""Tests for tensor product smooth (te/ti/t2) implementation.

Validates:
- Tensor product basis shapes
- Kronecker sum penalty shapes
- Row-wise Kronecker product
- te() vs ti() differences
- Integration with GAM formula API
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymgcv.smooth.tensor_product import (
    TensorProductSmooth,
    TensorProductT2,
    _row_kron,
    _kron_sum_penalties,
)


# ── Utility ──────────────────────────────────────────────────────────────────

def make_2d_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(x1 * 2 * np.pi) * np.cos(x2 * 2 * np.pi) + rng.normal(0, 0.1, n)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})


# ── Row-wise Kronecker ────────────────────────────────────────────────────────

def test_row_kron_shape():
    """Row-wise kron should produce (n, k1*k2) matrices."""
    A = np.random.randn(50, 6)
    B = np.random.randn(50, 8)
    result = _row_kron(A, B)
    assert result.shape == (50, 48), f"Expected (50, 48), got {result.shape}"


def test_row_kron_first_row():
    """First row of row-kron should equal kron(A[0,:], B[0,:])."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    result = _row_kron(A, B)
    expected_row0 = np.kron(A[0, :], B[0, :])
    np.testing.assert_allclose(result[0, :], expected_row0)


def test_row_kron_last_row():
    """Last row of row-kron should equal kron(A[-1,:], B[-1,:])."""
    A = np.random.randn(10, 5)
    B = np.random.randn(10, 7)
    result = _row_kron(A, B)
    expected = np.kron(A[-1, :], B[-1, :])
    np.testing.assert_allclose(result[-1, :], expected)


# ── Kronecker Sum Penalties ───────────────────────────────────────────────────

def test_kron_sum_penalties_shapes():
    """Kronecker-sum penalties should have correct shapes."""
    P1 = np.eye(5)
    P2 = np.eye(8)
    penalties = _kron_sum_penalties([P1, P2])
    assert len(penalties) == 2
    assert penalties[0].shape == (40, 40)  # 5*8 × 5*8
    assert penalties[1].shape == (40, 40)


def test_kron_sum_penalties_three_margins():
    """Three-margin tensor should have 3 penalties of shape (k1*k2*k3)²."""
    P1 = np.eye(3)
    P2 = np.eye(4)
    P3 = np.eye(5)
    penalties = _kron_sum_penalties([P1, P2, P3])
    assert len(penalties) == 3
    total = 3 * 4 * 5
    for P in penalties:
        assert P.shape == (total, total)


def test_kron_sum_identity_sum():
    """For identity penalties, kron sum should equal block-identity times ndim."""
    P1 = np.eye(4)
    P2 = np.eye(4)
    penalties = _kron_sum_penalties([P1, P2])
    # P1⊗I + I⊗P2 = I⊗I + I⊗I = 2*I for identity margins
    total = penalties[0] + penalties[1]
    np.testing.assert_allclose(total, 2 * np.eye(16), atol=1e-12)


# ── TensorProductSmooth: te() ─────────────────────────────────────────────────

def test_te_basis_shape():
    """te() basis should be (n, k1*k2)."""
    df = make_2d_data(80)
    tps = TensorProductSmooth(
        {'x1': df['x1'].values, 'x2': df['x2'].values},
        var_names=['x1', 'x2'],
        k_values=[8, 8],
    )
    assert tps.B.shape == (80, 64), f"Expected (80, 64), got {tps.B.shape}"


def test_te_penalty_count():
    """te() should return one penalty per margin variable."""
    df = make_2d_data(80)
    tps = TensorProductSmooth(
        {'x1': df['x1'].values, 'x2': df['x2'].values},
        var_names=['x1', 'x2'],
        k_values=[6, 7],
    )
    pens = tps.penalty_matrices()
    assert len(pens) == 2, f"Expected 2 penalties, got {len(pens)}"
    assert pens[0].shape == (42, 42)  # 6*7


def test_te_three_variables():
    """te() with 3 variables should give basis (n, k1*k2*k3)."""
    rng = np.random.default_rng(0)
    n = 50
    data = {'x1': rng.normal(size=n), 'x2': rng.normal(size=n), 'x3': rng.normal(size=n)}
    tps = TensorProductSmooth(data, ['x1', 'x2', 'x3'], k_values=[4, 4, 4])
    assert tps.B.shape == (n, 64)
    assert len(tps.penalty_matrices()) == 3


def test_te_total_dim():
    """total_dim attribute should match basis columns."""
    df = make_2d_data()
    tps = TensorProductSmooth(
        {'x1': df['x1'].values, 'x2': df['x2'].values},
        ['x1', 'x2'], k_values=[5, 6]
    )
    assert tps.total_dim == 30
    assert tps.B.shape[1] == tps.total_dim


def test_te_basis_finite():
    """All te() basis values should be finite (no NaN/inf)."""
    df = make_2d_data()
    tps = TensorProductSmooth(
        {'x1': df['x1'].values, 'x2': df['x2'].values},
        ['x1', 'x2'], k_values=[8, 8]
    )
    assert np.all(np.isfinite(tps.B)), "Basis contains non-finite values"


def test_te_penalty_symmetric():
    """Kronecker-sum penalties should be symmetric."""
    df = make_2d_data(60)
    tps = TensorProductSmooth(
        {'x1': df['x1'].values, 'x2': df['x2'].values},
        ['x1', 'x2'], k_values=[5, 5]
    )
    for P in tps.penalty_matrices():
        np.testing.assert_allclose(P, P.T, atol=1e-12, err_msg="Penalty not symmetric")


# ── TensorProductSmooth: ti() ─────────────────────────────────────────────────

def test_ti_basis_shape():
    """ti() basis should have same shape as te()."""
    df = make_2d_data(80)
    ti = TensorProductSmooth(
        {'x1': df['x1'].values, 'x2': df['x2'].values},
        ['x1', 'x2'], k_values=[7, 7], interaction_only=True
    )
    te = TensorProductSmooth(
        {'x1': df['x1'].values, 'x2': df['x2'].values},
        ['x1', 'x2'], k_values=[7, 7], interaction_only=False
    )
    assert ti.B.shape == te.B.shape


def test_ti_different_from_te():
    """ti() and te() should produce different bases."""
    df = make_2d_data(80)
    data = {'x1': df['x1'].values, 'x2': df['x2'].values}
    ti = TensorProductSmooth(data, ['x1', 'x2'], k_values=[6, 6], interaction_only=True)
    te = TensorProductSmooth(data, ['x1', 'x2'], k_values=[6, 6], interaction_only=False)
    # They should not be identical
    assert not np.allclose(ti.B, te.B), "ti and te should produce different bases"


# ── TensorProductT2 ───────────────────────────────────────────────────────────

def test_t2_basis_same_as_te():
    """t2() should have same basis matrix shape as te()."""
    df = make_2d_data(60)
    data = {'x1': df['x1'].values, 'x2': df['x2'].values}
    t2 = TensorProductT2(data, ['x1', 'x2'], k_values=[5, 5])
    te = TensorProductSmooth(data, ['x1', 'x2'], k_values=[5, 5])
    assert t2.B.shape == te.B.shape


# ── GAM Formula API with te/ti ────────────────────────────────────────────────

def test_gam_te_fits():
    """GAM with te() smooth should fit without errors."""
    from pymgcv.api.gam import GAM
    df = make_2d_data(80)
    model = GAM('y ~ te(x1, x2)', data=df, family='gaussian')
    model.fit(max_outer_iter=3, max_inner_iter=10, use_gpu=False)
    assert model.fitted
    assert model.beta is not None
    assert np.all(np.isfinite(model.beta))


def test_gam_te_predictions():
    """GAM predictions with te() should be finite."""
    from pymgcv.api.gam import GAM
    df = make_2d_data(80)
    model = GAM('y ~ te(x1, x2)', data=df, family='gaussian')
    model.fit(max_outer_iter=3, max_inner_iter=10, use_gpu=False)
    preds = model.predict(df)
    assert preds.shape == (80,)
    assert np.all(np.isfinite(preds))


def test_gam_ti_fits():
    """GAM with ti() interaction smooth should fit without errors."""
    from pymgcv.api.gam import GAM
    df = make_2d_data(60)
    model = GAM('y ~ ti(x1, x2)', data=df, family='gaussian')
    model.fit(max_outer_iter=3, max_inner_iter=10, use_gpu=False)
    assert model.fitted
    assert np.all(np.isfinite(model.beta))


def test_gam_te_edf_positive():
    """te() EDF should be positive after fitting."""
    from pymgcv.api.gam import GAM
    df = make_2d_data(80)
    model = GAM('y ~ te(x1, x2)', data=df, family='gaussian')
    model.fit(max_outer_iter=3, max_inner_iter=10, use_gpu=False)
    assert model.edf is not None
    assert model.edf > 0


def test_gam_te_poisson():
    """te() smooth should work with Poisson family."""
    from pymgcv.api.gam import GAM
    rng = np.random.default_rng(42)
    n = 80
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    log_mu = 1.0 + 0.5 * np.sin(x1 * 3) + 0.3 * np.cos(x2 * 3)
    y = rng.poisson(np.exp(log_mu)).astype(float)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    model = GAM('y ~ te(x1, x2)', data=df, family='poisson')
    model.fit(max_outer_iter=3, max_inner_iter=15, use_gpu=False)
    assert model.fitted


def test_formula_parser_te():
    """Formula parser should correctly parse te(x1, x2)."""
    from pymgcv.utils.formula_parser import FormulaParser
    parser = FormulaParser('y ~ te(x1, x2)')
    assert len(parser.smooth_terms) == 1
    spec = parser.smooth_terms[0]
    assert spec.term_type == 'te'
    assert 'x1' in spec.variables
    assert 'x2' in spec.variables


def test_formula_parser_ti():
    """Formula parser should correctly parse ti(x1, x2)."""
    from pymgcv.utils.formula_parser import FormulaParser
    parser = FormulaParser('y ~ ti(x1, x2)')
    assert len(parser.smooth_terms) == 1
    spec = parser.smooth_terms[0]
    assert spec.term_type == 'ti'


def test_formula_parser_t2():
    """Formula parser should correctly parse t2(x1, x2)."""
    from pymgcv.utils.formula_parser import FormulaParser
    parser = FormulaParser('y ~ t2(x1, x2)')
    assert len(parser.smooth_terms) == 1
    spec = parser.smooth_terms[0]
    assert spec.term_type == 't2'


def test_model_matrix_te_shape():
    """ModelMatrix with te() should give correct design matrix shape."""
    from pymgcv.utils.model_matrix import ModelMatrix
    df = make_2d_data(50)
    # te(x1, x2) with k_values default [10,10] → 100 tensor cols + 1 intercept
    mm = ModelMatrix(df, 'y ~ te(x1, x2)', center=False)
    # Intercept(1) + te basis(k1*k2)  
    assert mm.X.shape[0] == 50
    assert mm.X.shape[1] > 1  # at least intercept + basis cols
    assert len(mm.smooth_bases) == 1
    assert isinstance(mm.smooth_bases[0], TensorProductSmooth)
