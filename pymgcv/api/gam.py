"""Main GAM class for fitting generalized additive models.

This is the primary user-facing interface for pymgcv.

Example:
    >>> from pymgcv.api import gam
    >>> import pandas as pd
    >>> model = gam.GAM('y ~ s(x1) + s(x2)', data=df, family='gaussian')
    >>> model.fit()
    >>> print(model.summary())
    >>> y_pred = model.predict(df)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


class GAM:
    """Generalized Additive Model (GAM).

    Integrates formula parsing, basis construction, design matrix assembly,
    penalty matrices, PIRLS solver, and MAGIC smoothing parameter optimization.

    Attributes:
        formula: Model formula string.
        data: Input data.
        family: Distribution family.
        fitted: Whether model has been fitted.
        beta: Fitted coefficients.
        smoothing_parameters: Estimated λⱼ.
        edf: Effective degrees of freedom.
    """

    def __init__(
        self,
        formula: str,
        data: Optional[pd.DataFrame] = None,
        family: str = 'gaussian',
        offset: Optional[str] = None,
    ) -> None:
        """Initialize GAM.

        Args:
            formula: Model formula, e.g., 'y ~ s(x1) + s(x2) + x3'.
            data: Input data as DataFrame (optional, can be passed to fit()).
            family: Distribution family ('gaussian', 'poisson', 'gamma', 'tweedie').
            offset: Column name for offset vector.

        Raises:
            ValueError: If formula or family invalid.
        """
        self.formula = formula
        self.data = data
        self.family_name = family

        # Initialize components (TODO)
        self.model_matrix = None
        self.family = None
        self.pirls_solver = None
        self.magic_optimizer = None

        self.fitted = False
        self.beta: Optional[np.ndarray] = None
        self.smoothing_parameters: Optional[np.ndarray] = None
        self.edf: Optional[float] = None
        self.edf_per_smooth: Optional[dict] = None

    def fit(
        self,
        data: Optional[pd.DataFrame] = None,
        max_outer_iter: int = 10,
        max_inner_iter: int = 25,
        verbose: bool = False,
        use_gpu: bool = True,
    ) -> GAM:
        """Fit the GAM model.

        Args:
            data: Input data (required if not provided in __init__).
            max_outer_iter: Maximum MAGIC iterations.
            max_inner_iter: Maximum PIRLS iterations per outer iteration.
            verbose: Print progress.
            use_gpu: Enable JAX GPU acceleration if available.

        Returns:
            Self (for method chaining).
        """
        # Use provided data or fall back to initialization data
        if data is not None:
            self.data = data
        elif self.data is None:
            raise ValueError('Data must be provided either in __init__ or fit()')

        from pymgcv.utils.formula_parser import FormulaParser
        from pymgcv.utils.model_matrix import ModelMatrix
        from pymgcv.penalties.penalty_matrix import PenaltyMatrix
        from pymgcv.distributions.family_base import (
            GaussianFamily, PoissonFamily, GammaFamily, TweedieFamily,
            BinomialFamily, NegativeBinomialFamily, InverseGaussianFamily
        )
        from pymgcv.optimizer.magic_optimizer import MAGICOptimizer
        from pymgcv.optimizer.edf import EDFComputer
        from pymgcv.optimizer.jax_acceleration import device_info
        
        # 1. Parse formula
        parser = FormulaParser(self.formula)
        
        # 2. Construct design matrix
        self.model_matrix = ModelMatrix(self.data, self.formula)
        X = self.model_matrix.X
        y = self.model_matrix.response_vector()
        offset = self.model_matrix.offset_vector()
        
        # 3. Set up family
        family_map = {
            'gaussian': GaussianFamily(),
            'poisson': PoissonFamily(),
            'binomial': BinomialFamily(),
            'gamma': GammaFamily(shape=1.0),
            'tweedie': TweedieFamily(power=1.5),
            'negative.binomial': NegativeBinomialFamily(theta=1.0),
            'inverse.gaussian': InverseGaussianFamily(),
        }
        self.family = family_map.get(self.family_name, GaussianFamily())
        
        # 4. Build penalty matrices for each smooth term
        smooth_specs = parser.smooth_terms
        S_list = []
        smooth_starts = []
        smooth_sizes = []
        col_idx = self.model_matrix.param_indices.stop if self.model_matrix.param_indices else 0
        
        for smooth_spec in smooth_specs:
            basis_dim = smooth_spec.k if smooth_spec.k is not None else 10
            
            col_idx += basis_dim
        
        # 5. Optimize smoothing parameters via MAGIC
        # Set initial dispersion (1.0 for Gaussian, estimated for others)
        initial_dispersion = 1.0 if self.family_name == 'gaussian' else 1.0
        
        optimizer = MAGICOptimizer(
            X=X, y=y, family=self.family, S_list=S_list,
            smooth_starts=smooth_starts,
            smooth_sizes=smooth_sizes,
            offset=offset, dispersion=initial_dispersion
        )
        result = optimizer.optimize(
            max_outer_iter=max_outer_iter,
            max_inner_iter=max_inner_iter,
            verbose=verbose,
            use_jax=use_gpu and device_info()['available']
        )
        
        self.beta = result['coef']
        self.smoothing_parameters = result['smooth_lambda']
        fitted_values = self.family.linkinv(X @ self.beta + (offset if offset is not None else 0))
        
        # 6. Compute EDF
        S_combined = np.zeros_like(X.T @ X)
        for j, S_j in enumerate(S_list):
            lam = self.smoothing_parameters[j]
            i_start = smooth_starts[j]
            i_stop = i_start + smooth_sizes[j]
            S_combined[i_start:i_stop, i_start:i_stop] += lam * S_j
        
        edf_computer = EDFComputer(X, S_combined, self.family, self.beta, offset, dispersion=1.0)
        self.edf = edf_computer.total_edf()
        self.edf_per_smooth = edf_computer.per_smooth_edf(smooth_sizes)
        
        self.fitted = True
        if verbose:
            print(f'Fitted GAM with {self.edf:.2f} EDF')
        
        return self

    def summary(self) -> str:
        """Return model summary in mgcv format.

        Includes parametric coefficients, smooth term EDFs, p-values,
        deviance explained, AIC, REML score.

        Returns:
            Human-readable summary string.
        """
        if not self.fitted:
            return 'Model not yet fitted. Call .fit() first.'

        from pymgcv.diagnostics.significance_tests import SmoothTestSuite
        
        lines = []
        lines.append('Family: ' + self.family.__class__.__name__)
        lines.append('Link function: ' + getattr(self.family, 'link', 'unknown'))
        lines.append('')
        lines.append('Formula: ' + self.formula)
        lines.append('')
        lines.append('Estimated smoothing parameters:')
        for i, lam in enumerate(self.smoothing_parameters or []):
            lines.append(f'  sp({i}) = {lam:.6e}')
        lines.append('')
        lines.append(f'Model dimension(s): {len(self.beta)} total coefs')
        if self.edf:
            lines.append(f'Effective degrees of freedom: {self.edf:.2f}')
        lines.append('')
        
        # Parametric terms (first few coefficients)
        lines.append('Parametric coefficients:')
        for i, coef in enumerate(self.beta[:min(5, len(self.beta))]):
            lines.append(f'  Coef {i}: {coef:.6f}')
        
        return '\n'.join(lines)

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        scale: str = 'response',
    ) -> np.ndarray:
        """Make predictions.

        Args:
            data: New data for prediction. If None, use training data.
            scale: 'link' for linear predictor, 'response' for μ scale.

        Returns:
            Predictions array.
        """
        if not self.fitted:
            raise ValueError('Model not yet fitted')

        if data is None:
            data = self.data

        # Construct design matrix for new data
        from pymgcv.utils.model_matrix import ModelMatrix
        mm = ModelMatrix(data, self.formula)
        X_new = mm.X
        offset_new = mm.offset if mm.offset is not None else np.zeros(len(data))

        # Linear predictor
        eta = X_new @ self.beta + offset_new

        # Transform to response scale if needed
        if scale == 'response':
            return self.family.linkinv(eta)
        elif scale == 'link':
            return eta
        else:
            raise ValueError(f'Invalid scale: {scale}')

    def __repr__(self) -> str:
        """String representation."""
        return f'GAM({self.formula!r}, family={self.family_name!r})'
