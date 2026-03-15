"""Residual diagnostics and computations.

Computes:
- Deviance residuals
- Pearson residuals
- Standardized residuals
- QQ plot data
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from pymgcv.api.gam import GAM
from pymgcv.distributions.family_base import Family


def compute_residuals(
    y: np.ndarray,
    mu: np.ndarray,
    family: Family | None = None,
    type: str = 'deviance',
) -> np.ndarray:
    """Compute residuals.

    Args:
        y: Observations.
        mu: Predicted mean.
        family: Distribution family (for deviance residuals).
        type: 'deviance', 'pearson', 'response', or 'standardized'.

    Returns:
        Residuals vector.

    References:
        - McCullagh & Nelder (1989): Generalized Linear Models
        - Wood (2017): Generalized Additive Models
    """
    if type == 'response':
        # Observed - Fitted
        return y - mu

    elif type == 'pearson':
        # (y - μ) / √Var(Y)
        if family is None:
            variance = mu
        else:
            variance = family.variance(mu)
        return (y - mu) / np.sqrt(variance + 1e-10)

    elif type == 'deviance':
        # ±√(2*(logL(y) - logL(μ)))
        if family is None:
            # Fallback for Gaussian
            return (y - mu)
        else:
            # Deviance residuals
            sign = np.sign(y - mu)
            loglik_sat = family.loglik(y, y)
            loglik_fitted = family.loglik(y, mu)
            deviance = 2 * (loglik_sat - loglik_fitted)
            return sign * np.sqrt(np.abs(deviance))

    elif type == 'standardized':
        # Pearson residuals divided by √(1 - hat_diag)
        pearson = compute_residuals(y, mu, family, type='pearson')
        # Approximation: use (1 - 0.5) as crude leverage estimate
        return pearson / np.sqrt(0.5 + 1e-10)

    else:
        raise ValueError(f'Unknown residual type: {type}')


class ResidualDiagnostics:
    """Residual analysis for fitted GAM.

    Attributes:
        model: Fitted GAM.
        residuals: Dict of residual types.
        fitted: Fitted values.
        standardized_residuals: Standardized residuals.
    """

    def __init__(self, model: GAM) -> None:
        """Initialize residual diagnostics.

        Args:
            model: Fitted GAM model.
        """
        if not model.fitted:
            raise RuntimeError('Model not fitted')

        self.model = model
        self.y = model.data[model.formula.split('~')[0].strip()].values
        self.fitted = model.predict(model.data, scale='response')

        # Compute different residual types
        self.residuals = {
            'response': compute_residuals(self.y, self.fitted, model.family, 'response'),
            'pearson': compute_residuals(self.y, self.fitted, model.family, 'pearson'),
            'deviance': compute_residuals(self.y, self.fitted, model.family, 'deviance'),
            'standardized': compute_residuals(self.y, self.fitted, model.family, 'standardized'),
        }

    def summary(self) -> str:
        """Return residual summary.

        Returns:
            String with residual statistics.
        """
        lines = [
            'Residual Diagnostics',
            '====================',
        ]

        for res_type, res_vals in self.residuals.items():
            lines.append(f'\n{res_type.capitalize()} Residuals:')
            lines.append(f'  Min: {np.min(res_vals):.6f}')
            lines.append(f'  Q1:  {np.percentile(res_vals, 25):.6f}')
            lines.append(f'  Med: {np.median(res_vals):.6f}')
            lines.append(f'  Q3:  {np.percentile(res_vals, 75):.6f}')
            lines.append(f'  Max: {np.max(res_vals):.6f}')

            # Normality test
            stat, pval = stats.shapiro(res_vals[:min(5000, len(res_vals))])
            lines.append(f'  Shapiro-Wilk p-value: {pval:.6f}')

        return '\n'.join(lines)

    def get_residuals(self, type: str = 'deviance') -> np.ndarray:
        """Get residuals of specific type.

        Args:
            type: 'response', 'pearson', 'deviance', or 'standardized'.

        Returns:
            Residuals array.
        """
        return self.residuals.get(type, self.residuals['deviance'])

    def qq_plot_data(self, type: str = 'deviance') -> tuple:
        """Get Q-Q plot data.

        Args:
            type: Residual type.

        Returns:
            (theoretical_quantiles, sample_quantiles) tuple.
        """
        residuals = self.get_residuals(type)
        theoretical = stats.norm.ppf(
            np.linspace(0, 1, len(residuals) + 2)[1:-1]
        )
        sample = np.sort(residuals)
        return theoretical, sample

    def scale_location_data(self, type: str = 'pearson') -> tuple:
        """Get scale-location plot data.

        Args:
            type: Residual type.

        Returns:
            (fitted, sqrt_std_residuals) tuple.
        """
        residuals = self.get_residuals(type)
        std_residuals = residuals / np.std(residuals + 1e-10)
        sqrt_std = np.sqrt(np.abs(std_residuals))
        return self.fitted, sqrt_std
