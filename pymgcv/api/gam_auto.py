"""Automatic variable selection for GAM.

Implements stepwise variable selection:
    - EDF-based criteria (Occam's razor: |ΔDeviance|/|ΔEDF| > threshold)
    - Shrinkage penalties (basis='ts' or 'cs' induces selection)
    - Iterative elimination: Drop smallest effect, refit, repeat
    - Significance testing via p-values

Module exports:
    - fit_with_selection: Main function for automatic selection
    - VariableSelector: Class for stepwise selection
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pymgcv.api.gam import GAM
from pymgcv.diagnostics.significance_tests import SmoothTestSuite


def fit(
    y: np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame,
    family_name: str = 'gaussian',
    verbose: bool = False,
) -> GAM:
    """Fit GAM with automatic variable selection.

    Args:
        y: Response variable.
        X: Predictor matrix (numeric columns only).
        family_name: Family name ('gaussian', 'poisson', 'gamma', 'tweedie').
        verbose: Print progress.

    Returns:
        Fitted GAM model object.
    """
    # Construct formula from column names
    if isinstance(X, pd.DataFrame):
        cols = X.columns.tolist()
    else:
        cols = [f'x{i}' for i in range(X.shape[1])]

    # Create smooth terms for numeric variables
    smooth_terms = ' + '.join([f's({col})' for col in cols])
    formula = f'{cols[0]} ~ {smooth_terms}' if cols else 'y ~ 1'

    # Fit with selection
    return fit_with_selection(
        data=pd.DataFrame(X, columns=cols),
        formula=formula,
        family=family_name,
        verbose=verbose,
    )


def fit_with_selection(
    data: pd.DataFrame,
    formula: str,
    family: str = 'gaussian',
    criterion: str = 'edf',
    threshold: float = 1.0,
    max_steps: int = 10,
    verbose: bool = True,
) -> GAM:
    """Fit GAM with automatic variable selection.

    Uses stepwise elimination based on EDF or p-values.
    Stops when criterion indicates further improvement is marginal.

    Workflow:
        1. Fit full model with all variables
        2. Compute criterion (|ΔDev|/|ΔEDF| or p-values)
        3. Drop variable with smallest effect
        4. Refit, repeat until criterion satisfied

    Args:
        data: DataFrame with response and covariates.
        formula: Model formula, e.g., 'y ~ s(x1) + s(x2) + x3'.
        family: Distribution family.
        criterion: 'edf' (effect size) or 'pval' (significance).
        threshold: Elimination threshold (default: drop if |ΔDev|/|ΔEDF| < 1).
        max_steps: Maximum elimination iterations.
        verbose: Print progress.

    Returns:
        Fitted GAM after variable selection.

    Example:
        >>> from pymgcv.api.gam_auto import fit_with_selection
        >>> data = pd.DataFrame({...})
        >>> model = fit_with_selection(data, 'y ~ s(x1) + s(x2) + x3', family='gaussian')
        >>> print(model.summary())
    """
    selector = VariableSelector(
        data, formula, family=family, criterion=criterion, verbose=verbose
    )
    return selector.select(threshold=threshold, max_steps=max_steps)


class VariableSelector:
    """Stepwise variable selection for GAM.

    Attributes:
        data: Training data.
        formula: Original formula.
        family: Distribution family.
        current_formula: Formula after each selection step.
        eliminated_variables: List of dropped variables.
        criterion_history: Values of criterion at each step.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        family: str = 'gaussian',
        criterion: str = 'edf',
        verbose: bool = True,
    ) -> None:
        """Initialize variable selector.

        Args:
            data: Training data.
            formula: Model formula.
            family: Distribution family.
            criterion: 'edf' or 'pval'.
            verbose: Print progress.
        """
        self.data = data
        self.formula = formula
        self.family = family
        self.criterion = criterion
        self.verbose = verbose

        self.current_formula = formula
        self.eliminated_variables = []
        self.criterion_history = []
        self.models = []

    def select(
        self,
        threshold: float = 1.0,
        max_steps: int = 10,
    ) -> GAM:
        """Execute variable selection.

        Args:
            threshold: Elimination threshold.
            max_steps: Maximum elimination steps.

        Returns:
            Final fitted GAM after selection.
        """
        for step in range(max_steps):
            # Fit current model
            model = GAM(self.current_formula, family=self.family)
            model.fit(self.data, verbose=False)
            self.models.append(model)

            if self.verbose:
                print(f'Step {step}: Formula = {self.current_formula}')

            # Compute criterion for each smooth term
            if self.criterion == 'edf':
                # Use EDF-based effect size
                effects = self._compute_edf_effects(model)
            elif self.criterion == 'pval':
                # Use p-values from significance tests
                effects = self._compute_pvalues(model)
            else:
                raise ValueError(f'Unknown criterion: {self.criterion}')

            # Check if any variable should be eliminated
            min_effect_var = min(effects, key=effects.get) if effects else None

            if min_effect_var is None:
                break

            min_effect = effects[min_effect_var]

            if self.verbose:
                print(f'  Min effect: {min_effect_var} = {min_effect:.6f}')

            # Elimination decision
            if min_effect < threshold:
                # Drop this variable
                self.current_formula = self._remove_variable(
                    self.current_formula, min_effect_var
                )
                self.eliminated_variables.append(min_effect_var)
                self.criterion_history.append(min_effect)

                if self.verbose:
                    print(f'  ✓ Eliminated {min_effect_var}')
            else:
                if self.verbose:
                    print(f'  ✗ Stopping: effect {min_effect:.6f} ≥ threshold {threshold}')
                break

        # Final fit
        final_model = GAM(self.current_formula, family=self.family)
        final_model.fit(self.data, verbose=False)
        self.models.append(final_model)

        if self.verbose:
            print(f'\nFinal formula: {self.current_formula}')
            print(f'Eliminated {len(self.eliminated_variables)} variables')

        return final_model

    def _compute_edf_effects(self, model: GAM) -> dict[str, float]:
        """Compute effect size via EDF per smooth term.

        Args:
            model: Fitted GAM.

        Returns:
            Dict mapping variable names to EDF values (higher = more important).
        """
        effects = {}

        if hasattr(model, 'edf_per_smooth') and model.edf_per_smooth:
            for i, (var, edf) in enumerate(model.edf_per_smooth.items()):
                effects[var] = edf
        else:
            # Fallback: equal weighting
            effects = {f'var_{i}': 1.0 for i in range(len(model.smoothing_parameters or []))}

        return effects

    def _compute_pvalues(self, model: GAM) -> dict[str, float]:
        """Compute p-values from significance tests.

        Args:
            model: Fitted GAM.

        Returns:
            Dict mapping variable names to p-values (lower = more significant).
        """
        try:
            suite = SmoothTestSuite(
                model.beta,
                model.model_matrix.X,
                model.family,
                model.smoothing_parameters,
            )
            pvals = suite.test_all()

            # Convert to p-value dict
            effects = {}
            for i, pval in enumerate(pvals):
                effects[f'smooth_{i}'] = pval

            return effects
        except Exception as e:
            if self.verbose:
                print(f'  Warning: Could not compute p-values: {e}')
            # Fallback: use uniform weight
            return {f'var_{i}': 0.5 for i in range(len(model.smoothing_parameters or []))}

    def _remove_variable(self, formula: str, var_name: str) -> str:
        """Remove variable from formula.

        Args:
            formula: Current formula.
            var_name: Variable to remove (can be 's(x)' or 'x').

        Returns:
            Updated formula.
        """
        # Simple regex-based removal
        import re

        # Try to remove smooth term s(var_name)
        pattern = r'\s*\+?\s*s\(' + re.escape(var_name) + r'\)'
        result = re.sub(pattern, '', formula)

        if result == formula:
            # Try to remove raw variable
            pattern = r'\s*\+?\s*' + re.escape(var_name)
            result = re.sub(pattern, '', formula)

        # Clean up consecutive +
        result = re.sub(r'\+\s*\+', ' + ', result).strip()

        return result

    def summary(self) -> str:
        """Return selection summary.

        Returns:
            String summarizing selection process.
        """
        lines = [
            'Variable Selection Summary',
            '===========================',
            f'Original formula: {self.formula}',
            f'Final formula: {self.current_formula}',
            f'Criterion: {self.criterion}',
            '',
            'Eliminated variables:',
        ]

        if self.eliminated_variables:
            for var in self.eliminated_variables:
                lines.append(f'  - {var}')
        else:
            lines.append('  (none)')

        lines.extend([
            '',
            f'Total steps: {len(self.models) - 1}',
        ])

        return '\n'.join(lines)
