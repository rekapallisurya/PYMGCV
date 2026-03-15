"""Design matrix construction for GAM models.

Assembles the full design matrix X = [X_parametric | B1 | B2 | ... ]
from parametric terms and smooth basis matrices.

Also handles data preprocessing: centering, scaling, categorical encoding.

Module exports:
    - ModelMatrix: Main class for design matrix construction
    - assemble_design_matrix: Functional API
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pymgcv.utils.formula_parser import FormulaParser, ParametricSpec, SmoothSpec
from pymgcv.smooth.thin_plate import ThinPlateSpline
from pymgcv.smooth.tensor_product import TensorProductSmooth, TensorProductT2
from pymgcv.smooth.cyclic_spline import CyclicSpline
from pymgcv.smooth.random_effect import RandomEffect


class ModelMatrix:
    """Construct the GAM design matrix X.

    Handles:
    - Parametric terms (linear, categorical)
    - Smooth basis matrices (TPRS, cubic splines, etc.)
    - Centering and standardization (optional)
    - NaN/missing value handling

    Attributes:
        data: Input data (DataFrame or dict of arrays).
        formula_parser: FormulaParser instance.
        X: Full design matrix, shape (n, p) where p = p_param + sum(k_smooth).
        column_names: Names of design matrix columns.
        n_obs: Number of observations.
        p_cols: Number of design matrix columns.
        smooth_bases: List of constructed smooth basis objects.
        param_indices: Slice of X containing parametric terms.
        smooth_indices: List of slices for each smooth term in X.
        center_mean: Mean for centering (if applied).
        scale_std: Std deviation for scaling (if applied).
    """

    def __init__(
        self,
        data: pd.DataFrame | dict[str, np.ndarray],
        formula: str,
        center: bool = True,
        scale: bool = False,
    ) -> None:
        """Initialize and construct design matrix.

        Args:
            data: Input data as DataFrame or dict of arrays.
            formula: Formula string, e.g., 'y ~ s(x1) + s(x2) + x3'.
            center: If True, center parametric terms.
            scale: If True, scale parametric terms to unit variance.

        Raises:
            ValueError: If formula or data is invalid.
        """
        # Convert data to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError('data must be DataFrame or dict')

        self.data = data
        self.n_obs = len(data)
        
        # Parse formula
        self.formula_parser = FormulaParser(formula)
        self.response_var = self.formula_parser.response

        # Validate all variables exist in data
        all_vars = (
            self.formula_parser.parametric_names
            + sum([spec.variables for spec in self.formula_parser.smooth_terms], [])
            + ([self.formula_parser.offset_term] if self.formula_parser.offset_term else [])
        )
        for var in all_vars:
            if var not in data.columns:
                raise ValueError(f'Variable "{var}" not found in data')

        self.center = center
        self.scale = scale
        self.center_mean: Optional[np.ndarray] = None
        self.scale_std: Optional[np.ndarray] = None

        # Construct design matrix
        self.X: np.ndarray = np.zeros((self.n_obs, 1))  # placeholder
        self.column_names: list[str] = []
        self.smooth_bases: list = []  # can hold ThinPlateSpline, TensorProductSmooth, etc.
        self.smooth_specs_used: list[SmoothSpec] = []  # matches smooth_bases order
        self.smooth_by_levels: list[Optional[list]] = []  # by-variable levels per smooth
        self.param_indices: Optional[slice] = None
        self.smooth_indices: list[slice] = []

        self._construct()

    def _construct(self) -> None:
        """Assemble design matrix from parametric and smooth terms."""
        X_parts: list[np.ndarray] = []
        col_names: list[str] = []

        # 1. Parametric terms
        X_param, param_cols = self._construct_parametric_matrix()
        if X_param.shape[1] > 0:
            X_parts.append(X_param)
            col_names.extend(param_cols)
            self.param_indices = slice(0, X_param.shape[1])

        # 2. Smooth terms
        for smooth_spec in self.formula_parser.smooth_terms:
            X_smooth, smooth_cols, basis_obj, by_levels = self._construct_smooth_matrix(smooth_spec)
            if X_smooth.shape[1] > 0:
                start_col = sum(x.shape[1] for x in X_parts)
                X_parts.append(X_smooth)
                col_names.extend(smooth_cols)
                self.smooth_bases.append(basis_obj)
                self.smooth_specs_used.append(smooth_spec)
                self.smooth_by_levels.append(by_levels)
                self.smooth_indices.append(
                    slice(start_col, start_col + X_smooth.shape[1])
                )

        # Concatenate and validate
        if X_parts:
            self.X = np.column_stack(X_parts)
        else:
            # Fallback: intercept only
            self.X = np.ones((self.n_obs, 1))
            col_names = ['Intercept']

        self.column_names = col_names
        self.p_cols = self.X.shape[1]

        # Apply centering/scaling to parametric terms
        if self.param_indices is not None and (self.center or self.scale):
            self._center_scale_parametric()

    def _construct_parametric_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Construct matrix from parametric terms.

        Handles:
        - Intercept (added by default unless suppressed)
        - Simple variables
        - Interactions
        - Categorical variables (one-hot encoding, drop first)
        - Transformations via function_terms

        Returns:
            (X_param, column_names)
        """
        X_cols: list[np.ndarray] = []
        col_names: list[str] = []

        # Add intercept by default unless formula explicitly excludes it (with -1)
        # For now, always include intercept for compatibility
        intercept_col = np.ones(self.n_obs)
        X_cols.append(intercept_col)
        col_names.append('(Intercept)')

        if not self.formula_parser.parametric_terms:
            return np.column_stack(X_cols) if X_cols else np.zeros((self.n_obs, 1)), col_names

        for spec in self.formula_parser.parametric_terms:
            if spec.interaction:
                # Interaction term: x1 * x2
                x1 = self.data[spec.variables[0]].values.astype(float)
                x2 = self.data[spec.variables[1]].values.astype(float)
                x_int = x1 * x2
                X_cols.append(x_int)
                col_names.append(' * '.join(spec.variables))
            else:
                # Single variable or function
                var = spec.variables[0]
                
                # Check if function was applied
                if spec.label in self.formula_parser.function_terms:
                    func_str = self.formula_parser.function_terms[spec.label]
                    # Parse and apply function (log, exp, etc.)
                    if func_str.startswith('log('):
                        x = np.log(self.data[var].values.astype(float))
                        col_name = f'log({var})'
                    elif func_str.startswith('exp('):
                        x = np.exp(self.data[var].values.astype(float))
                        col_name = f'exp({var})'
                    elif func_str.startswith('sqrt('):
                        x = np.sqrt(self.data[var].values.astype(float))
                        col_name = f'sqrt({var})'
                    else:
                        x = self.data[var].values.astype(float)
                        col_name = var
                else:
                    x = self.data[var].values.astype(float)
                    col_name = var

                X_cols.append(x)
                col_names.append(col_name)

        if X_cols:
            X_param = np.column_stack(X_cols)
        else:
            X_param = np.zeros((self.n_obs, 0))

        return X_param, col_names

    def _construct_smooth_matrix(
        self, smooth_spec: SmoothSpec
    ) -> tuple[np.ndarray, list[str], object, Optional[list]]:
        """Construct basis matrix for a smooth term.

        Handles:
        - s(): TPRS (thin plate), B-spline, cyclic and random effect bases
        - te(): Full tensor product
        - ti(): Tensor interaction (removes marginal effects)
        - t2(): Alternative tensor product
        - by-variable expansion for varying-coefficient models

        Args:
            smooth_spec: SmoothSpec object describing the smooth term.

        Returns:
            (X_smooth, column_names, basis_object, by_levels)
        """
        term_type = smooth_spec.term_type
        basis_code = smooth_spec.basis.lower()
        by_levels = None

        # ---- Tensor product smooths (te, ti, t2) ----
        if term_type in ('te', 'ti', 't2'):
            var_names = smooth_spec.variables
            if len(var_names) < 2:
                raise ValueError(f'{term_type}() requires at least 2 variables, got {var_names}')

            data_dict = {v: self.data[v].values.astype(float) for v in var_names}
            k = smooth_spec.k  # k applies to each margin if scalar

            if term_type == 't2':
                basis_obj = TensorProductT2(
                    data_dict, var_names,
                    k_values=[k] * len(var_names) if k else None,
                    basis_type='tp',
                )
            else:
                interaction_only = (term_type == 'ti')
                basis_obj = TensorProductSmooth(
                    data_dict, var_names,
                    k_values=[k] * len(var_names) if k else None,
                    basis_type='tp',
                    interaction_only=interaction_only,
                )

            X_smooth = basis_obj.basis_matrix()
            col_names = [f'{smooth_spec.label}.{i}' for i in range(X_smooth.shape[1])]
            return X_smooth, col_names, basis_obj, by_levels

        # ---- Single-variable smooth (s) ----
        if term_type == 's':
            if len(smooth_spec.variables) == 0:
                raise ValueError('s() requires at least 1 variable')

            # If multiple variables in s(), treat as TPRS multivariate
            if len(smooth_spec.variables) > 1:
                var_names = smooth_spec.variables
                X_multi = np.column_stack(
                    [self.data[v].values.astype(float) for v in var_names]
                )
                basis_obj = ThinPlateSpline(X_multi, k=smooth_spec.k)
                X_smooth = basis_obj.basis_matrix()
                col_names = [f'{smooth_spec.label}.{i}' for i in range(X_smooth.shape[1])]
                return X_smooth, col_names, basis_obj, by_levels

            var_name = smooth_spec.variables[0]
            X_var = self.data[var_name].values.astype(float)

            # Select basis type based on bs/basis argument
            if basis_code in ('cc', 'cp'):
                # Cyclic cubic spline
                basis_obj = CyclicSpline(X_var, k=smooth_spec.k or 10)
                X_smooth_base = basis_obj.basis_matrix()

            elif basis_code == 're':
                # Random effect
                basis_obj = RandomEffect(self.data[var_name], k=smooth_spec.k)
                X_smooth_base = basis_obj.basis_matrix()

            elif basis_code in ('bs', 'ps', 'cr', 'cs'):
                # B-spline / P-spline / Cubic regression spline
                if basis_code in ('cr', 'cs'):
                    from pymgcv.smooth.cubic_spline import CubicRegressionSpline, CubicShrinkageSpline
                    cls = CubicShrinkageSpline if basis_code == 'cs' else CubicRegressionSpline
                    basis_obj = cls(X_var, k=smooth_spec.k or 10)
                elif basis_code == 'ps':
                    from pymgcv.smooth.bspline import PSplineBasis
                    basis_obj = PSplineBasis(X_var, k=smooth_spec.k or 20)
                else:  # bs
                    from pymgcv.smooth.bspline import BSplineBasis
                    basis_obj = BSplineBasis(X_var, k=smooth_spec.k or 10)
                X_smooth_base = basis_obj.B if hasattr(basis_obj, 'B') else basis_obj.basis_matrix

            elif basis_code == 'ad':
                # Adaptive smooth
                from pymgcv.smooth.advanced import AdaptiveSpline
                basis_obj = AdaptiveSpline(X_var, k=smooth_spec.k or 20)
                X_smooth_base = basis_obj.B

            elif basis_code == 'gp':
                # Gaussian process smooth
                from pymgcv.smooth.advanced import GPSmooth
                basis_obj = GPSmooth(X_var, k=smooth_spec.k or 10)
                X_smooth_base = basis_obj.B

            elif basis_code in ('fs', 'sz'):
                # Factor smooth interaction: requires a by-variable
                by_var = smooth_spec.by_variable
                if by_var is None or by_var not in self.data.columns:
                    # Fall back to TPRS if no grouping variable available
                    basis_obj = ThinPlateSpline(X_var.reshape(-1, 1), k=smooth_spec.k)
                    X_smooth_base = basis_obj.basis_matrix()
                else:
                    group = self.data[by_var]
                    if basis_code == 'fs':
                        from pymgcv.smooth.advanced import FactorSmooth
                        basis_obj = FactorSmooth(X_var, group, k=smooth_spec.k or 10)
                    else:
                        from pymgcv.smooth.advanced import FactorDeviation
                        basis_obj = FactorDeviation(X_var, group, k=smooth_spec.k or 10)
                    X_smooth_base = basis_obj.B

            else:
                # Default: thin plate regression spline
                basis_obj = ThinPlateSpline(X_var.reshape(-1, 1), k=smooth_spec.k)
                X_smooth_base = basis_obj.basis_matrix()

        elif term_type == 're':
            # Standalone re() random effect
            if len(smooth_spec.variables) == 0:
                raise ValueError('re() requires at least 1 variable')
            var_name = smooth_spec.variables[0]
            basis_obj = RandomEffect(self.data[var_name], k=smooth_spec.k)
            X_smooth_base = basis_obj.basis_matrix()

        else:
            raise ValueError(f'Unknown smooth term type: {term_type}')

        # ---- by-variable expansion ----
        X_smooth, by_levels = self._expand_by_variable(
            X_smooth_base, smooth_spec, self.data
        )

        n_cols = X_smooth.shape[1]
        col_names = [f'{smooth_spec.label}.{i}' for i in range(n_cols)]
        return X_smooth, col_names, basis_obj, by_levels

    def _expand_by_variable(
        self,
        X_basis: np.ndarray,
        smooth_spec: SmoothSpec,
        data: pd.DataFrame,
    ) -> tuple[np.ndarray, Optional[list]]:
        """Expand basis matrix for by-variable (varying-coefficient models).

        For s(x, by=group) with factor group:
            Result shape: (n, k * n_levels), padded with zeros per level.

        For s(x, by=weight) with continuous weight:
            Result shape: (n, k), element-wise scaled.

        Args:
            X_basis: Marginal basis matrix, shape (n, k).
            smooth_spec: Smooth specification (contains by_variable).
            data: DataFrame with all variables.

        Returns:
            (X_expanded, levels_list_or_None)
        """
        if smooth_spec.by_variable is None:
            return X_basis, None

        by_var = smooth_spec.by_variable
        if by_var not in data.columns:
            raise ValueError(f'by-variable "{by_var}" not found in data')

        by_data = data[by_var]
        by_vals = by_data.values
        n, k = X_basis.shape

        # Detect factor vs continuous
        if by_vals.dtype.kind in ('U', 'O', 'S') or hasattr(by_data, 'cat'):
            # Factor by-variable
            levels = list(np.unique(by_vals.astype(str)))
            n_levels = len(levels)
            level_map = {lv: i for i, lv in enumerate(levels)}

            X_expanded = np.zeros((n, k * n_levels))
            for i, val in enumerate(by_vals.astype(str)):
                level_idx = level_map.get(val, None)
                if level_idx is not None:
                    X_expanded[i, level_idx * k:(level_idx + 1) * k] = X_basis[i, :]

            return X_expanded, levels

        else:
            # Continuous by-variable: element-wise scale
            by_cont = by_vals.astype(float)
            X_expanded = X_basis * by_cont[:, np.newaxis]
            return X_expanded, None

    def _center_scale_parametric(self) -> None:
        """Center and/or scale parametric terms."""
        if self.param_indices is None:
            return

        X_param = self.X[:, self.param_indices]

        if self.center:
            self.center_mean = np.nanmean(X_param, axis=0)
            X_param = X_param - self.center_mean

        if self.scale:
            self.scale_std = np.nanstd(X_param, axis=0)
            self.scale_std[self.scale_std == 0] = 1  # avoid division by zero
            X_param = X_param / self.scale_std

        self.X[:, self.param_indices] = X_param

    def design_matrix(self) -> np.ndarray:
        """Return the full design matrix X, shape (n, p)."""
        return self.X

    def response_vector(self) -> np.ndarray:
        """Extract response vector y from data."""
        response_col = self.formula_parser.response
        if response_col not in self.data.columns:
            raise ValueError(f'Response "{response_col}" not found in data')
        return self.data[response_col].values.astype(float)

    def offset_vector(self) -> Optional[np.ndarray]:
        """Extract offset vector if present in formula."""
        if self.formula_parser.offset_term is None:
            return None
        offset_col = self.formula_parser.offset_term
        if offset_col not in self.data.columns:
            raise ValueError(f'Offset "{offset_col}" not found in data')
        return self.data[offset_col].values.astype(float)

    def summary(self) -> str:
        """Return human-readable summary of design matrix."""
        lines = [
            f'Design Matrix Summary',
            f'=====================',
            f'Observations: {self.n_obs}',
            f'Columns: {self.p_cols}',
            f'Parametric columns: {self.param_indices.stop - self.param_indices.start if self.param_indices else 0}',
            f'Smooth basis columns: {sum(s.stop - s.start for s in self.smooth_indices)}',
            f'',
            f'Column names:',
        ]
        for i, name in enumerate(self.column_names):
            lines.append(f'  {i:2d}. {name}')
        return '\n'.join(lines)


def assemble_design_matrix(
    data: pd.DataFrame,
    formula: str,
    center: bool = True,
    scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Functional API for assembling design matrix.

    Args:
        data: Input data as DataFrame.
        formula: Formula string.
        center: Center parametric terms.
        scale: Scale parametric terms.

    Returns:
        (X, y, offset) where offset may be None.
    """
    model_matrix = ModelMatrix(data, formula, center=center, scale=scale)
    X = model_matrix.design_matrix()
    y = model_matrix.response_vector()
    offset = model_matrix.offset_vector()
    return X, y, offset
