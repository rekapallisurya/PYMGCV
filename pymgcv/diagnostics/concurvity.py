"""Concurvity diagnostics.

Measures collinearity (concurvity) among smooth terms.
High concurvity indicates smooth terms are highly correlated,
making coefficient estimates unstable.

References:
    - Wood (2017): Generalized Additive Models, Section 4.8
    - Yee & Mitchell (1991): Additive models for categorical data
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from pymgcv.api.gam import GAM


def concurvity(X_smooth: np.ndarray, indices: list[slice]) -> dict:
    """Compute concurvity indices.

    Concurvity = 1 - 1/(condition number of correlation matrix).

    Args:
        X_smooth: Design matrix columns for smooth terms.
        indices: List of slices for each smooth term.

    Returns:
        Dict with concurvity statistics per smooth.
    """
    results = {}

    # Extract basis matrices for each smooth
    bases = [X_smooth[:, idx] for idx in indices]

    # Compute concurvity matrix
    n_smooths = len(bases)
    concurv_matrix = np.zeros((n_smooths, n_smooths))

    for i in range(n_smooths):
        for j in range(n_smooths):
            if i == j:
                concurv_matrix[i, j] = 1.0
            else:
                # Concurvity between smooths i and j
                # Use canonical correlation or correlation of fitted values
                try:
                    # Compute correlation between predictions from smooths
                    pred_i = bases[i]
                    pred_j = bases[j]

                    # Normalize
                    pred_i_norm = pred_i / (np.linalg.norm(pred_i, axis=0) + 1e-10)
                    pred_j_norm = pred_j / (np.linalg.norm(pred_j, axis=0) + 1e-10)

                    # Maximum absolute correlation
                    corr = np.abs(np.max(np.corrcoef(pred_i_norm.T, pred_j_norm.T)))
                    concurv_matrix[i, j] = np.clip(corr, 0, 1)
                except:
                    concurv_matrix[i, j] = 0.0

    # Overall concurvity (condition number)
    try:
        eigenvals = np.linalg.eigvals(concurv_matrix)
        condition_number = np.max(eigenvals) / np.clip(np.min(eigenvals), 1e-10, None)
        overall_concurv = 1 - 1 / np.sqrt(condition_number)
    except:
        overall_concurv = 0.0

    results['concurvity_matrix'] = concurv_matrix
    results['overall'] = np.clip(overall_concurv, 0, 1)
    results['pairwise'] = {
        (i, j): concurv_matrix[i, j]
        for i in range(n_smooths)
        for j in range(i + 1, n_smooths)
    }

    return results


class ConcurvityDiagnostics:
    """Compute concurvity statistics for fitted GAM.

    Attributes:
        model: Fitted GAM.
        concurvity_result: Dict with concurvity statistics.
        pairwise_concurv: Pairwise concurvity indices.
        overall_concurv: Overall concurvity measure.
    """

    def __init__(self, model: GAM, threshold: float = 0.8) -> None:
        """Initialize concurvity diagnostics.

        Args:
            model: Fitted GAM.
            threshold: Concurvity threshold for warning.
        """
        if not model.fitted:
            raise RuntimeError('Model not fitted')

        self.model = model
        self.threshold = threshold

        # Extract smooth basis matrix (design matrix minus parametric cols)
        X = model.model_matrix.X
        n_parametric = model.model_matrix.n_parametric if hasattr(model.model_matrix, 'n_parametric') else 0

        X_smooth = X[:, n_parametric:]

        # Construct indices for each smooth term
        if hasattr(model, 'smoothing_parameters'):
            n_smooths = len(model.smoothing_parameters)
            basis_dim_per_smooth = X_smooth.shape[1] // max(1, n_smooths)
            indices = [
                slice(i * basis_dim_per_smooth, (i + 1) * basis_dim_per_smooth)
                for i in range(n_smooths)
            ]
        else:
            indices = [slice(0, X_smooth.shape[1])]

        # Compute concurvity
        self.concurvity_result = concurvity(X_smooth, indices)
        self.pairwise_concurv = self.concurvity_result.get('pairwise', {})
        self.overall_concurv = self.concurvity_result.get('overall', 0.0)

        # Flag high concurvity pairs
        self.high_concurv_pairs = [
            (i, j) for (i, j), v in self.pairwise_concurv.items()
            if v > threshold
        ]

    def summary(self) -> str:
        """Return concurvity summary.

        Returns:
            String summary.
        """
        lines = [
            'Concurvity Diagnostics',
            '======================',
            '',
            f'Overall concurvity: {self.overall_concurv:.4f}',
            '',
        ]

        if self.high_concurv_pairs:
            lines.append('High concurvity pairs (threshold > 0.8):')
            for i, j in self.high_concurv_pairs:
                lines.append(
                    f'  Smooth_{i} × Smooth_{j}: {self.pairwise_concurv.get((i, j), 0):.4f}'
                )
        else:
            lines.append('No problematic concurvity detected.')

        lines.extend([
            '',
            'Interpretation:',
            '  - Concurvity > 0.8: High collinearity, estimates unstable',
            '  - Concurvity < 0.5: Low collinearity, comfortable',
            '  - Overall > 0.9: Serious multicollinearity problem',
        ])

        return '\n'.join(lines)

    def has_problems(self, threshold: float | None = None) -> bool:
        """Check if concurvity indicates problems.

        Args:
            threshold: Custom threshold (default: 0.8).

        Returns:
            True if problematic concurvity detected.
        """
        thresh = threshold or self.threshold
        return (
            self.overall_concurv > 0.95 or
            any(v > thresh for v in self.pairwise_concurv.values())
        )
