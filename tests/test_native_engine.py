"""Tests for native math backends (C extension + Fortran LAPACK path)."""

from __future__ import annotations

import numpy as np

from pymgcv.linalg.native_engine import (
    backend_info,
    col_squared_norms,
    solve_spd_fortran,
)


def test_backend_info_shape() -> None:
    info = backend_info()
    assert "c_engine" in info
    assert "fortran_lapack" in info
    assert isinstance(info["c_engine"], bool)
    assert isinstance(info["fortran_lapack"], bool)


def test_col_squared_norms_matches_numpy() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(size=(30, 12))
    got = col_squared_norms(a)
    expect = np.sum(a * a, axis=0)
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-12)


def test_solve_spd_fortran_matches_numpy() -> None:
    rng = np.random.default_rng(123)
    m = rng.normal(size=(20, 20))
    a = m.T @ m + 1e-3 * np.eye(20)
    b = rng.normal(size=20)

    got = solve_spd_fortran(a, b)
    expect = np.linalg.solve(a, b)
    np.testing.assert_allclose(got, expect, rtol=1e-10, atol=1e-10)
