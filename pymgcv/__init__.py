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
from pymgcv import config
from pymgcv.api import gam, gam_auto, predict, summary
from pymgcv.api.bam import BAM, bam
from pymgcv.api.gam import GAM
from pymgcv.api.gam_check import gam_check, k_check
from pymgcv.api.gamm import GAMM, gamm
from pymgcv.api.model_comparison import aic, anova_gam, bic, compare_models
from pymgcv.api.plot import plot_diagnostics, plot_gam, plot_residuals, plot_smooth, plot_smooth_2d
from pymgcv.distributions.family_base import TweedieFamily as _TweedieFamily


def s(
    *variables: str,
    k: int | None = None,
    bs: str = "tp",
    by: str | None = None,
    fx: bool = False,
    m: int | None = None,
) -> str:
    """Return a smooth-term string for use inside a GAM formula.

    Example::

        GAM(f"y ~ {s('x1')} + {s('x2', k=10, bs='cr')}", data=df)
    """
    args = ", ".join(variables)
    if bs != "tp":
        args += f", bs='{bs}'"
    if k is not None:
        args += f", k={k}"
    if by is not None:
        args += f", by={by}"
    if fx:
        args += ", fx=True"
    if m is not None:
        args += f", m={m}"
    return f"s({args})"


class Tweedie(_TweedieFamily):
    """Tweedie family convenience alias supporting an optional ``link`` keyword.

    ``link`` must be ``"log"`` (the only supported link); it is accepted for
    API compatibility and ignored otherwise.

    Args:
        p: Variance power parameter (1 < p < 2). Default 1.5.
        power: Alias for ``p``.
        link: Must be ``"log"`` (only supported link).
        estimate_power: If True, estimate p from data during fitting
            (mirrors R's ``tw()``). Default False.

    Example::

        GAM("y ~ s(x)", family=Tweedie(link="log"), ...)          # fixed p=1.5
        GAM("y ~ s(x)", family=Tweedie(estimate_power=True), ...)  # estimate p like tw()
    """

    def __init__(
        self,
        p: float = 1.5,
        power: float | None = None,
        link: str = "log",
        estimate_power: bool = False,
    ) -> None:
        if link != "log":
            raise ValueError(f"Tweedie only supports link='log', got '{link}'")
        super().__init__(power=power if power is not None else p, estimate_power=estimate_power)


__all__ = [
    # Main model classes
    "GAM",
    "BAM",
    "GAMM",
    # Convenience functions
    "bam",
    "gamm",
    # Formula helpers
    "s",
    "Tweedie",
    # Model comparison
    "anova_gam",
    "compare_models",
    "aic",
    "bic",
    # Plotting
    "plot_smooth",
    "plot_smooth_2d",
    "plot_residuals",
    "plot_diagnostics",
    "plot_gam",
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
