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
        self.smooth_bases: list[ThinPlateSpline] = []
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
            X_smooth, smooth_cols, basis_obj = self._construct_smooth_matrix(smooth_spec)
            if X_smooth.shape[1] > 0:
                start_col = sum(x.shape[1] for x in X_parts)
                X_parts.append(X_smooth)
                col_names.extend(smooth_cols)
                self.smooth_bases.append(basis_obj)
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
        - Simple variables
        - Interactions
        - Categorical variables (one-hot encoding, drop first)
        - Transformations via function_terms

        Returns:
            (X_param, column_names)
        """
        if not self.formula_parser.parametric_terms:
            return np.zeros((self.n_obs, 0)), []

        X_cols: list[np.ndarray] = []
        col_names: list[str] = []

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
    ) -> tuple[np.ndarray, list[str], ThinPlateSpline]:
        """Construct basis matrix for a smooth term.

        Args:
            smooth_spec: SmoothSpec object describing the smooth term.

        Returns:
            (X_smooth, column_names, basis_object)
        """
        # For now, implement TPRS (thin plate regression splines)
        # TODO: Add support for other bases (cubic spline, etc.)

        if smooth_spec.term_type == 's':
            # Single-variable smooth
            if len(smooth_spec.variables) != 1:
                raise ValueError(
                    f'Single smooth s() expects 1 variable, got {len(smooth_spec.variables)}'
                )
            
            var_name = smooth_spec.variables[0]
            X_var = self.data[var_name].values.reshape(-1, 1).astype(float)
            
            # Construct TPRS basis
            basis = ThinPlateSpline(X_var, k=smooth_spec.k)
            X_smooth = basis.basis_matrix()
            
        elif smooth_spec.term_type == 'te':
            # Tensor product smooth
            # TODO: Implement tensor product basis
            raise NotImplementedError('Tensor product (te) not yet implemented')
        else:
            raise ValueError(f'Unknown smooth term type: {smooth_spec.term_type}')

        # Column names for this smooth term
        col_names = [f'{smooth_spec.label}.{i}' for i in range(X_smooth.shape[1])]

        return X_smooth, col_names, basis

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
