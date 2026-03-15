"""
pymgcv: Production-grade GAM package with numerical equivalence to R's mgcv.

This package provides a complete implementation of Generalized Additive Models (GAMs)
achieving numerical equivalence with Simon Wood's R package mgcv within tolerance 1e-6.

Core exports:
    - GAM:          Main generalized additive model class (gam)
    - BAM:          Fast GAM for large datasets (bam)
    - GAMM:         Generalized additive mixed model (gamm)
    - anova_gam:    ANOVA-style model comparison
    - compare_models: AIC/BIC comparison table
    - aic, bic:     Information criteria
    - gam_check:    Diagnostic checks and k-adequacy test
    - k_check:      k-index adequacy test
    - gam_auto.fit: Automatic variable selection for GAMs
    - smooth_basis:  Smooth term basis functions
    - predict:       Prediction interface
    - summary:       Model summaries

Smooth basis types supported:
    tp, tprs        Thin plate regression spline (default)
    cr, cs          Cubic regression / shrinkage spline
    bs              B-spline
    ps              P-spline (Eilers-Marx)
    cc, cp          Cyclic cubic / P-spline
    re              Random effect
    gp              Gaussian process smooth
    ad              Adaptive smooth
    fs              Factor smooth interaction
    sz              Smooth deviation by factor
    te, ti, t2      Tensor product smooths

Example:
    >>> from pymgcv import GAM, BAM, GAMM, anova_gam, gam_check
    >>> import pandas as pd
    >>> model = GAM('y ~ s(x)', data=df)
    >>> model.fit()
    >>> print(model.summary())
    >>> gam_check(model)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Surya"

# API entry points
from pymgcv.api import gam, gam_auto, predict, summary
from pymgcv.api.gam import GAM
from pymgcv.api.bam import BAM, bam
from pymgcv.api.gamm import GAMM, gamm
from pymgcv.api.model_comparison import anova_gam, compare_models, aic, bic
from pymgcv.api.gam_check import gam_check, k_check
from pymgcv import config

__all__ = [
    # Main model classes
    "GAM",
    "BAM",
    "GAMM",
    # Convenience functions
    "bam",
    "gamm",
    # Model comparison
    "anova_gam",
    "compare_models",
    "aic",
    "bic",
    # Diagnostics
    "gam_check",
    "k_check",
    # Modules
    "config",
    "gam",
    "gam_auto",
    "predict",
    "summary",
]

