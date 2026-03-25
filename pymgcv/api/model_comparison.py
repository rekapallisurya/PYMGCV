"""Model comparison tools for GAMs.

Implements:
    - anova_gam(): LRT/F-test comparison of nested GAMs
    - compare_models(): AIC/deviance comparison table
    - AIC/BIC for GAM objects

References:
    - Wood, S.N. (2017). GAMs: An Introduction with R, §6.12.
    - Hastie, T. & Tibshirani, R. (1990). Generalized Additive Models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import f as f_dist


def _get_fitted_info(model) -> dict:
    """Extract fitted model information.

    Returns dict with: n, edf, deviance, loglik, dispersion, formula.
    """
    info = {}
    info["formula"] = getattr(model, "formula", "?")
    info["edf"] = float(getattr(model, "edf", 0) or 0)
    info["dispersion"] = float(getattr(model, "dispersion_", 1.0))

    X = getattr(model, "_X_fit", None)
    y = getattr(model, "_y_fit", None)
    beta = getattr(model, "beta", None)
    family = getattr(model, "family", None)

    if X is not None and y is not None and beta is not None and family is not None:

        offset = getattr(model, "model_matrix", None)
        offset_vec = (
            offset.offset_vector()
            if (offset is not None and hasattr(offset, "offset_vector"))
            else np.zeros(len(y))
        )
        if offset_vec is None:
            offset_vec = np.zeros(len(y))
        eta = X @ beta + offset_vec
        mu = family.linkinv(eta)
        info["n"] = len(y)
        info["loglik"] = float(family.loglik(y, mu, info["dispersion"]))
        info["deviance"] = float(-2 * info["loglik"])
    else:
        info["n"] = 0
        info["loglik"] = np.nan
        info["deviance"] = np.nan

    return info


def aic(model, k: float = 2.0) -> float:
    """Compute AIC for a fitted GAM.

    AIC = -2 * loglik + k * edf

    Args:
        model: Fitted GAM/BAM/GAMM object.
        k: Penalty per parameter (default 2 for AIC, log(n) for BIC).

    Returns:
        AIC value (lower is better).
    """
    info = _get_fitted_info(model)
    return -2 * info["loglik"] + k * info["edf"]


def bic(model) -> float:
    """Compute BIC for a fitted GAM.

    BIC = -2 * loglik + log(n) * edf

    Args:
        model: Fitted GAM/BAM/GAMM object.

    Returns:
        BIC value.
    """
    info = _get_fitted_info(model)
    n = max(info["n"], 1)
    return aic(model, k=np.log(n))


def anova_gam(*models, test: str = "F") -> pd.DataFrame:
    """ANOVA-style comparison of fitted GAMs.

    Performs sequential LRT comparing nested models from simplest to
    most complex. Models must be in order from simplest to most complex.

    Two test types:
        - 'F':  F-test (appropriate when dispersion is estimated)
                F = (ΔDeviance / Δdf) / φ̂  ~  F(Δdf, n - edf_complex)
        - 'Chisq': Chi-squared LRT (appropriate when dispersion is known)
                X² = ΔDeviance / φ̂  ~  χ²(Δdf)

    Args:
        *models: Two or more fitted GAM objects (in order simple → complex).
        test: Test statistic ('F' or 'Chisq').

    Returns:
        DataFrame with columns: Model, Df, Deviance, Resid.Df, Resid.Dev,
        [Df, Deviance, F/Chisq, p-value] for each pairwise comparison.

    Raises:
        ValueError: If fewer than 2 models provided or test is invalid.

    Example:
        >>> m1 = GAM('y ~ s(x1)', data=df).fit()
        >>> m2 = GAM('y ~ s(x1) + s(x2)', data=df).fit()
        >>> print(anova_gam(m1, m2))
    """
    if len(models) < 2:
        raise ValueError("anova_gam requires at least 2 models")
    if test not in ("F", "Chisq"):
        raise ValueError("test must be 'F' or 'Chisq'")

    infos = [_get_fitted_info(m) for m in models]

    rows = []
    for i, (info, model) in enumerate(zip(infos, models)):
        row = {
            "Model": info["formula"],
            "Resid.Df": max(info["n"] - info["edf"], 0.5),
            "Resid.Dev": info["deviance"],
            "Df": np.nan,
            "Deviance": np.nan,
            "stat": np.nan,
            "p-value": np.nan,
        }
        if i > 0:
            prev = infos[i - 1]
            delta_dev = prev["deviance"] - info["deviance"]
            delta_df = info["edf"] - prev["edf"]
            phi = info["dispersion"]

            if delta_df <= 0:
                row["Df"] = delta_df
                row["Deviance"] = delta_dev
            else:
                row["Df"] = delta_df
                row["Deviance"] = delta_dev
                if test == "F":
                    f_stat = (delta_dev / delta_df) / max(phi, 1e-10)
                    df2 = max(info["n"] - info["edf"], 1)
                    p_val = (
                        float(1 - f_dist.cdf(f_stat, delta_df, df2))
                        if np.isfinite(f_stat)
                        else np.nan
                    )
                    row["stat"] = f_stat
                    row["p-value"] = p_val
                else:  # Chisq
                    chi_stat = delta_dev / max(phi, 1e-10)
                    p_val = (
                        float(1 - chi2.cdf(chi_stat, delta_df)) if np.isfinite(chi_stat) else np.nan
                    )
                    row["stat"] = chi_stat
                    row["p-value"] = p_val

        rows.append(row)

    df = pd.DataFrame(rows)
    stat_col = "F" if test == "F" else "Chisq"
    df = df.rename(columns={"stat": stat_col})
    return df


def compare_models(*models, criterion: str = "AIC") -> pd.DataFrame:
    """Compare multiple fitted GAMs by information criterion.

    Args:
        *models: One or more fitted GAM objects.
        criterion: One of 'AIC', 'BIC', 'deviance'.

    Returns:
        DataFrame sorted by chosen criterion with columns:
        Model, n, EDF, Deviance, LogLik, AIC, BIC, [ΔAIC or ΔBIC].

    Example:
        >>> m1, m2 = GAM(...).fit(), GAM(...).fit()
        >>> print(compare_models(m1, m2))
    """
    if not models:
        raise ValueError("No models provided")

    rows = []
    for model in models:
        info = _get_fitted_info(model)
        a = aic(model)
        b = bic(model)
        rows.append(
            {
                "Formula": info["formula"],
                "n": info["n"],
                "EDF": round(info["edf"], 2),
                "Deviance": round(info["deviance"], 4),
                "LogLik": round(info["loglik"], 4),
                "AIC": round(a, 4),
                "BIC": round(b, 4),
            }
        )

    df = pd.DataFrame(rows)

    if criterion == "AIC":
        df = df.sort_values("AIC")
        best = df["AIC"].min()
        df["ΔAIC"] = (df["AIC"] - best).round(4)
    elif criterion == "BIC":
        df = df.sort_values("BIC")
        best = df["BIC"].min()
        df["ΔBIC"] = (df["BIC"] - best).round(4)
    elif criterion == "deviance":
        df = df.sort_values("Deviance")

    df = df.reset_index(drop=True)
    return df
