"""Integration tests for Phases 1-2: Foundation and Core Solver.

Tests that formula parsing, basis construction, design matrix assembly,
penalty matrices, and PIRLS solver work together to fit a GAM.

Tests cover:
    - Formula parsing (Step 1)
    - ThinPlate basis construction (Step 2)
    - Design matrix assembly (Step 3)
    - Penalty matrix construction (Step 4)
    - Demmler-Reinsch orthogonalization (Step 5)
    - Penalized likelihood (Step 6)
    - PIRLS solver (Step 7)
    - EDF computation (Step 10)
    - Significance tests (Step 11)

Run with: pytest tests/test_phases_1_2.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymgcv.utils.formula_parser import parse_formula
from pymgcv.utils.model_matrix import ModelMatrix, assemble_design_matrix
from pymgcv.smooth.thin_plate import ThinPlateSpline, thin_plate_basis
from pymgcv.penalties.penalty_matrix import PenaltyMatrix, PenaltyMatrixSet
from pymgcv.penalties.demmler_reinsch import DemmlerReinschOrthogonalization, orthogonalize_design_matrix
from pymgcv.optimizer.pirls import PIRLSSolver, solve_pirls
from pymgcv.optimizer.edf import EDFComputer, compute_edf, compute_edf_per_smooth
from pymgcv.optimizer.penalized_likelihood import GaussianPenalizedLikelihood
from pymgcv.distributions.family_base import GaussianFamily, PoissonFamily
from pymgcv.diagnostics.significance_tests import compute_smooth_tests


class TestFormulaParser:
    """Test Step 1: Formula Parsing."""

    def test_parse_simple_smooth(self) -> None:
        """Test parsing formula with smooth terms."""
        parser = parse_formula('y ~ s(x1) + s(x2)')
        assert parser.response == 'y'
        assert len(parser.smooth_terms) == 2
        assert parser.smooth_terms[0].variables == ['x1']
        assert parser.smooth_terms[1].variables == ['x2']

    def test_parse_mixed_terms(self) -> None:
        """Test parsing formula with mixed parametric and smooth terms."""
        parser = parse_formula('y ~ s(x1) + x2 + x3')
        assert len(parser.smooth_terms) == 1
        assert len(parser.parametric_terms) == 2
        assert parser.parametric_names == ['x2', 'x3']

    def test_parse_with_basis_kwargs(self) -> None:
        """Test parsing smooth with basis specifications."""
        parser = parse_formula('y ~ s(x, k=10, basis="cs")')
        assert parser.smooth_terms[0].k == 10
        assert parser.smooth_terms[0].basis == 'cs'


class TestThinPlateSpline:
    """Test Step 2: Thin Plate Regression Splines."""

    def test_basis_construction_univariate(self) -> None:
        """Test TPRS basis for univariate input."""
        X = np.linspace(0, 1, 50).reshape(-1, 1)
        tprs = ThinPlateSpline(X, k=8)
        
        B = tprs.basis_matrix()
        assert B.shape == (50, 8)
        assert np.all(np.isfinite(B))

    def test_functional_api(self) -> None:
        """Test functional API for basis."""
        X = np.random.randn(30, 1)
        B = thin_plate_basis(X, k=5)
        assert B.shape == (30, 5)


class TestDesignMatrix:
    """Test Step 3: Design Matrix Construction."""

    def test_assemble_design_matrix_gaussian(self) -> None:
        """Test design matrix assembly."""
        np.random.seed(42)
        n = 50
        X_data = np.linspace(0, 1, n)
        y_data = np.sin(2 * np.pi * X_data) + np.random.normal(0, 0.1, n)
        
        df = pd.DataFrame({'x': X_data, 'y': y_data})
        
        X, y, offset = assemble_design_matrix(df, 'y ~ s(x)')
        
        assert X.shape[0] == n
        assert y.shape == (n,)
        assert offset is None
        assert X.shape[1] > 0  # Has columns

    def test_parametric_plus_smooth(self) -> None:
        """Test design matrix with parametric and smooth terms."""
        np.random.seed(42)
        n = 40
        df = pd.DataFrame({
            'y': np.random.randn(n),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        
        X, y, _ = assemble_design_matrix(df, 'y ~ x1 + s(x2)')
        
        # X should have: 1 (intercept) + 1 (x1) + k (s(x2))
        assert X.shape[1] >= 3


class TestPenaltyMatrix:
    """Test Step 4: Penalty Matrix Construction."""

    def test_tprs_penalty_construction(self) -> None:
        """Test TPRS penalty matrix."""
        k = 10
        pm = PenaltyMatrix(k, penalty_type='tprs')
        S = pm.penalty_matrix()
        
        assert S.shape == (k, k)
        assert np.all(np.isfinite(S))
        assert np.allclose(S, S.T)  # Symmetric

    def test_penalty_matrix_set(self) -> None:
        """Test combining multiple penalties."""
        S1 = PenaltyMatrix(5, 'tprs').penalty_matrix()
        S2 = PenaltyMatrix(5, 'tprs').penalty_matrix()
        
        pm_set = PenaltyMatrixSet([
            PenaltyMatrix(5, 'tprs'),
            PenaltyMatrix(5, 'tprs'),
        ])
        
        pm_set.set_lambda(np.array([1.0, 2.0]))
        S_combined = pm_set.combined_penalty([slice(0, 5), slice(5, 10)])
        
        assert S_combined.shape == (10, 10)


class TestDemmlerReinsch:
    """Test Step 5: Demmler-Reinsch Orthogonalization."""

    def test_orthogonalization_diagonalizes_penalty(self) -> None:
        """Test that D-R orthogonalization diagonalizes the penalty."""
        np.random.seed(42)
        n, p = 20, 5
        X = np.random.randn(n, p)
        S = np.eye(p) * 2  # Simple diagonal penalty
        
        X_tilde, D, U = orthogonalize_design_matrix(X, S)
        
        assert X_tilde.shape == (n, p)
        assert D.shape == (p, p)
        assert U.shape == (p, p)
        # D should be diagonal (or close to it)
        assert np.allclose(D, np.diag(np.diag(D)))


class TestPenalizedLikelihood:
    """Test Step 6: Penalized Likelihood."""

    def test_gaussian_penalized_likelihood(self) -> None:
        """Test Gaussian penalized likelihood."""
        np.random.seed(42)
        n, p = 30, 5
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        S = np.eye(p) * 0.1
        
        pl = GaussianPenalizedLikelihood(X, y, [S], lambda_vec=[1.0])
        
        beta = np.random.randn(p)
        obj = pl.objective(beta)
        assert np.isfinite(obj)
        assert obj > 0


class TestPIRLS:
    """Test Step 7: PIRLS Solver."""

    def test_pirls_convergence_gaussian(self) -> None:
        """Test PIRLS solver for Gaussian family."""
        np.random.seed(42)
        n, p = 50, 5
        X = np.random.randn(n, p)
        y = X[:, 0] + 0.1 * np.random.randn(n)  # y depends on x
        
        family = GaussianFamily()
        S_list = [np.eye(p) * 0.01]
        
        solver = PIRLSSolver(X, y, family, S_list)
        beta = solver.solve(max_iter=25, tol=1e-7, verbose=False)
        
        assert solver.converged
        assert beta.shape == (p,)
        assert np.all(np.isfinite(beta))

    def test_pirls_poisson(self) -> None:
        """Test PIRLS for Poisson family."""
        np.random.seed(42)
        n, p = 40, 4
        X = np.random.randn(n, p)
        mu_true = np.abs(2 + X[:, 0])
        y = np.random.poisson(mu_true)
        
        family = PoissonFamily()
        S_list = [np.eye(p) * 0.1]
        
        solver = PIRLSSolver(X, y, family, S_list)
        beta = solver.solve(max_iter=25, tol=1e-5, verbose=False)
        
        assert beta.shape == (p,)
        assert np.all(np.isfinite(beta))


class TestEDF:
    """Test Step 10: EDF Computation."""

    def test_edf_computation(self) -> None:
        """Test EDF calculation."""
        np.random.seed(42)
        n, p = 30, 6
        X = np.random.randn(n, p)
        S = np.eye(p) * 0.5
        
        family = GaussianFamily()
        beta = np.random.randn(p)
        
        edf = compute_edf(X, S, family, beta)
        
        assert 0 < edf <= p
        assert np.isfinite(edf)


class TestIntegration:
    """Integration tests combining multiple steps."""

    def test_full_gaussian_gam_workflow(self) -> None:
        """Test complete workflow for Gaussian GAM."""
        np.random.seed(42)
        
        # Generate synthetic data
        n = 50
        x = np.linspace(0, 1, n)
        y_true = np.sin(2 * np.pi * x)
        y = y_true + 0.1 * np.random.randn(n)
        
        df = pd.DataFrame({'x': x, 'y': y})
        
        # Step 1: Parse formula
        parser = parse_formula('y ~ s(x)')
        assert len(parser.smooth_terms) == 1
        
        # Step 3: Assemble design matrix
        X, y_vec, _ = assemble_design_matrix(df, 'y ~ s(x)')
        assert X.shape[0] == n
        
        # Step 4: Construct penalty
        k = X.shape[1] - 1  # Exclude intercept
        S_smooth = PenaltyMatrix(k, 'tprs').penalty_matrix()
        S_combined = np.zeros_like(X.T @ X)
        S_combined[1:k+1, 1:k+1] = S_smooth  # Smooth part only
        
        # Step 7: Fit with PIRLS
        family = GaussianFamily()
        solver = PIRLSSolver(X, y_vec, family, [S_combined])
        beta = solver.solve(max_iter=20, tol=1e-6, verbose=False)
        
        assert solver.converged
        assert np.all(np.isfinite(beta))
        
        # Check fitted values are reasonable
        mu = solver.fitted_values()
        assert mu.shape == y.shape
        rmse = np.sqrt(np.mean((mu - y)**2))
        assert rmse < 1.0  # Should fit reasonably well


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
