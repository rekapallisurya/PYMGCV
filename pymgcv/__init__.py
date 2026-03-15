"""
pymgcv: Production-grade GAM package with numerical equivalence to R's mgcv.

This package provides a complete implementation of Generalized Additive Models (GAMs)
achieving numerical equivalence with Simon Wood's R package mgcv within tolerance 1e-6.

Core exports:
    - gam.GAM: Main generalized additive model class
    - gam_auto.fit: Automatic variable selection for GAMs
    - smooth_basis: Smooth term basis functions
    - predict: Prediction interface
    - summary: Model summaries

Example:
    >>> from pymgcv.api import gam
    >>> import pandas as pd
    >>> model = gam.GAM('y ~ s(x)', data=df)
    >>> model.fit()
    >>> print(model.summary())
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Surya"

# API entry points
from pymgcv.api import gam, gam_auto, predict, summary

__all__ = [
    "gam",
    "gam_auto",
    "predict",
    "summary",
]
