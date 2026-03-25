"""Influence diagnostics: leverage, cook's distance, DFBETAS.

References:
    - Wood (2017): Generalized Additive Models, Ch. 4
    - Fox & Weisberg (2011): Regression Diagnostics
"""

from __future__ import annotations

import numpy as np

from pymgcv.api.gam import GAM


def leverage(H: np.ndarray) -> np.ndarray:
    """Extract leverage (diagonal of hat matrix).

    Args:
        H: Hat matrix, shape (n, n).

    Returns:
        Leverage vector, shape (n,).
    """
    return np.diag(H)


def cooks_distance(
    residuals: np.ndarray,
    leverage_vals: np.ndarray,
    scale: float = 1.0,
    p: int | None = None,
) -> np.ndarray:
    """Compute Cook's distance.

    D_i = (r_i² / (p * s²)) * (h_i / (1 - h_i))

    Args:
        residuals: Residuals vector.
        leverage_vals: Leverage values (diagonal of hat matrix).
        scale: Estimated scale (dispersion).
        p: Number of parameters (default: None).

    Returns:
        Cook's distance vector.
    """
    if p is None:
        p = len(residuals)

    # Avoid division by zero
    denom = np.maximum(1.0 - leverage_vals, 1e-10)
    cd = (residuals**2 / (p * scale)) * (leverage_vals / denom)

    return cd


def dfbetas(
    X: np.ndarray,
    residuals: np.ndarray,
    leverage_vals: np.ndarray,
    XtX_inv: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Compute DFBETAS (change in coefficients when observation removed).

    DFBETAS_i = β̂_(-i) - β̂ ≈ (X'X)^-1 X_i r_i / (s(1 - h_i))

    Args:
        X: Design matrix.
        residuals: Residuals.
        leverage_vals: Leverage values.
        XtX_inv: Precomputed (X'X)^-1 (default: compute).
        scale: Estimated scale.

    Returns:
        DFBETAS matrix, shape (n, p).
    """
    n, p = X.shape

    if XtX_inv is None:
        XtX_inv = np.linalg.pinv(X.T @ X)

    # Avoid division by zero
    denom = np.maximum(1.0 - leverage_vals, 1e-10)

    # DFBETAS_i = (X'X)^-1 X_i r_i / (s(1 - h_i))
    dfbetas_mat = np.zeros((n, p))

    for i in range(n):
        dfbetas_mat[i, :] = XtX_inv @ X[i, :] * residuals[i] / (scale * denom[i])

    return dfbetas_mat


class InfluenceDiagnostics:
    """Compute influence statistics for fitted GAM.

    Attributes:
        model: Fitted GAM.
        leverage_vals: Leverage for each observation.
        cooks_d: Cook's distance for each observation.
        dfbetas_vals: DFBETAS matrix.
        influential_obs: List of influential observation indices.
    """

    def __init__(self, model: GAM, threshold: float = 0.05) -> None:
        """Initialize influence diagnostics.

        Args:
            model: Fitted GAM.
            threshold: Percentile threshold for flagging influential points.
        """
        if not model.fitted:
            raise RuntimeError("Model not fitted")

        self.model = model
        self.threshold = threshold

        # Extract components
        X = model.model_matrix.X
        y = model.data[model.formula.split("~")[0].strip()].values
        beta = model.beta
        fitted = model.predict(model.data, scale="response")
        residuals = y - fitted

        # Estimate scale (dispersion)
        self.scale = np.sum(residuals**2) / max(1, len(y) - len(beta))

        # Compute hat matrix diagonal (leverage)
        try:
            XtX_inv = np.linalg.inv(X.T @ X + 1e-8 * np.eye(X.shape[1]))
            self.leverage_vals = np.array([X[i, :] @ XtX_inv @ X[i, :] for i in range(len(y))])
        except:
            self.leverage_vals = np.ones(len(y)) / len(y)

        # Compute Cook's distance
        self.cooks_d = cooks_distance(residuals, self.leverage_vals, self.scale, len(beta))

        # Compute DFBETAS
        try:
            XtX_inv = np.linalg.inv(X.T @ X + 1e-8 * np.eye(X.shape[1]))
            self.dfbetas_vals = dfbetas(X, residuals, self.leverage_vals, XtX_inv, self.scale)
        except:
            self.dfbetas_vals = np.zeros((len(y), len(beta)))

        # Flag influential observations
        threshold_val = np.percentile(self.cooks_d, 100 * (1 - threshold))
        self.influential_obs = np.where(self.cooks_d > threshold_val)[0].tolist()

    def summary(self) -> str:
        """Return influence diagnostics summary.

        Returns:
            String summary.
        """
        lines = [
            "Influence Diagnostics",
            "====================",
            "",
            "Leverage (hat diagonal):",
            f"  Min: {np.min(self.leverage_vals):.6f}",
            f"  Mean: {np.mean(self.leverage_vals):.6f}",
            f"  Max: {np.max(self.leverage_vals):.6f}",
            "",
            "Cook's Distance:",
            f"  Min: {np.min(self.cooks_d):.6f}",
            f"  Mean: {np.mean(self.cooks_d):.6f}",
            f"  Max: {np.max(self.cooks_d):.6f}",
            "",
            f"Influential observations (top {len(self.influential_obs)}):",
        ]

        for idx in sorted(self.influential_obs)[:10]:
            lines.append(f"  Obs {idx}: Cook's D = {self.cooks_d[idx]:.6e}")

        return "\n".join(lines)

    def get_influential_threshold(self, p: float = 0.05) -> float:
        """Get threshold for influential observations.

        Args:
            p: Percentile (default: 0.05 = top 5%).

        Returns:
            Threshold value.
        """
        return np.percentile(self.cooks_d, 100 * (1 - p))
