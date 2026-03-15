"""Smoothing parameter management and MAGIC optimization."""

from __future__ import annotations

import numpy as np


class SmoothingParameterManager:
    """Manage smoothing parameters λⱼ across fitting."""

    def __init__(self, n_smooth: int, initial: float = 1.0):
        """Initialize.

        Args:
            n_smooth: Number of smooth terms.
            initial: Initial value for all λⱼ.
        """
        self.lambda_vec = np.full(n_smooth, initial, dtype=float)
        self.lambda_log = np.zeros(n_smooth)

    def set_log_lambda(self, log_lambda: np.ndarray) -> None:
        """Set smoothing parameters from log scale."""
        self.lambda_log = np.asarray(log_lambda, dtype=float)
        self.lambda_vec = np.exp(self.lambda_log)

    def smoothing_parameters(self) -> np.ndarray:
        """Return current smoothing parameters."""
        return self.lambda_vec.copy()
