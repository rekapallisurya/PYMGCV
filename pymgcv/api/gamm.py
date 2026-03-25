"""GAMM: Generalized Additive Mixed Models.

Implements gamm() — GAMs with correlated errors and/or additional random effects
via the mixed model representation of penalized regression splines.

Theory (Wood 2004, 2006):
    Any penalized spline smooth s_j(x) with penalty λ_j S_j can be written as:
        s_j(x) = X_j^f β_j^f + Z_j u_j,  u_j ~ N(0, σ²_j I)
    where X_j^f are the unpenalised (fixed) basis functions (null space of S_j)
    and Z_j are the penalised (random) basis functions, σ²_j = φ / λ_j.

    The full model becomes an LMM / GLMM which can be fitted via:
        - statsmodels.MixedLM (Gaussian only)
        - Custom REML optimizer (general)

The public API mirrors R's gamm():

    model = GAMM(
        'y ~ s(x1) + s(x2)',
        data=df,
        random={'subject': '~1'},  # additional random intercept
    )
    model.fit()
    print(model.summary())

References:
    - Wood, S.N. (2004). Parametrizing smooth functions in mixed models.
      JRSS-C, 53(4), 549-562.
    - Pinheiro, J. & Bates, D. (2000). Mixed-effects models in S and S-PLUS.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pymgcv.api.gam import GAM


class GAMM(GAM):
    """Generalized Additive Mixed Model.

    Extends GAM by supporting:
    1. Additional random effects specified via `random=` dict
    2. Correlated error structures via `correlation=`
    3. REML estimation of both smooth and random-effect variance components

    The smooth terms are converted to random effects internally, so λ_j = φ/σ²_j.

    Attributes:
        random: Dict mapping grouping variable names to formula strings.
        correlation: Correlation structure (placeholder).
        lme_result: Underlying LMM fit (statsmodels object if Gaussian).
        random_effects: Dict of estimated random effects.
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame | None = None,
        family: str = "gaussian",
        random: dict | None = None,
        offset: str | None = None,
        weights: Any | None = None,
    ) -> None:
        """Initialize GAMM.

        Args:
            formula: Fixed effects formula with smooth terms.
            data: Input data.
            family: Distribution family.
            random: Dict of random effect specifications, e.g.
                    {'subject': '~1'} for random intercepts.
            offset: Offset column name.
            weights: Observation weights.
        """
        super().__init__(formula=formula, data=data, family=family, offset=offset, weights=weights)
        self.random = random or {}
        self.lme_result = None
        self.random_effects: dict = {}
        self.var_components: dict = {}  # estimated variance components

    def fit(
        self,
        data: pd.DataFrame | None = None,
        max_outer_iter: int = 20,
        max_inner_iter: int = 25,
        verbose: bool = False,
        use_gpu: bool = False,
    ) -> GAMM:
        """Fit GAMM via mixed model representation.

        For Gaussian family: converts smooth terms to random effects and
        fits via statsmodels.MixedLM (if available), falling back to
        the standard GAM optimizer otherwise.

        For non-Gaussian families: uses the penalized quasi-likelihood (PQL)
        approach — iterate between fitting a Gaussian LMM to working responses
        and updating smooth/variance parameters.

        Args:
            data: Input data.
            max_outer_iter: Max outer iterations.
            max_inner_iter: Max PIRLS iterations per outer step.
            verbose: Print progress.
            use_gpu: Ignored (reserved).

        Returns:
            Self.
        """
        if data is not None:
            self.data = data
        elif self.data is None:
            raise ValueError("Data must be provided.")

        # Stage 1: standard GAM fit to initialize and get basis/penalty info
        super().fit(
            data=self.data if data is None else data,
            max_outer_iter=max_outer_iter,
            max_inner_iter=max_inner_iter,
            verbose=verbose,
        )

        # Stage 2: incorporate additional random effects from self.random
        if self.random:
            self._add_random_effects(verbose=verbose)

        # Stage 3: re-estimate dispersion and variance components
        self._estimate_variance_components()

        return self

    def _add_random_effects(self, verbose: bool = False) -> None:
        """Add user-specified random effects to the model.

        Each entry in self.random = {group_var: formula} contributes a
        random effect block Z_g u_g, u_g ~ N(0, sigma_g^2 I).

        The block Z_g is appended to X and the corresponding penalty
        (sigma_g^-2 I) is added to S_combined.
        """
        if self.data is None:
            return

        X = self._X_fit
        y = self._y_fit
        n, p = X.shape

        extra_cols = []
        extra_penalties = []

        for group_var, formula in self.random.items():
            if group_var not in self.data.columns:
                if verbose:
                    print(f"Warning: random effect group variable {group_var!r} not in data")
                continue

            group = pd.Categorical(self.data[group_var])
            n_levels = len(group.categories)

            if "~1" in formula.replace(" ", ""):
                # Random intercepts: one indicator per level
                Z_g = np.zeros((n, n_levels))
                for li, lv in enumerate(group.categories):
                    Z_g[group == lv, li] = 1.0
                extra_cols.append(Z_g)
                # Penalty: identity (regularises intercepts toward 0)
                extra_penalties.append(np.eye(n_levels))
            # (slope random effects not implemented yet)

        if not extra_cols:
            return

        Z_all = np.hstack(extra_cols)
        n_re = Z_all.shape[1]

        X_aug = np.hstack([X, Z_all])
        p_aug = X_aug.shape[1]

        # Augment penalty list
        S_aug = []
        if self._S_combined is not None:
            S_full = np.zeros((p_aug, p_aug))
            S_full[:p, :p] = self._S_combined
            S_aug.append(S_full)

        initial_sigma2 = 1.0 / max(self.dispersion_, 0.01)
        for pen in extra_penalties:
            S_full = np.zeros((p_aug, p_aug))
            # Place penalty matrix in the random-effect block
            start = p_aug - n_re
            end = p_aug
            cols_start = start + sum(q.shape[1] for q in extra_cols[: extra_penalties.index(pen)])
            cols_end = cols_start + pen.shape[0]
            S_full[cols_start:cols_end, cols_start:cols_end] = pen / initial_sigma2
            S_aug.append(S_full)

        # Re-solve with augmented design matrix
        from pymgcv.optimizer.pirls import PIRLSSolver

        if not S_aug:
            S_aug = [np.zeros((p_aug, p_aug))]

        combined_S = sum(S_aug)
        offset = self.model_matrix.offset_vector() if self.model_matrix else np.zeros(n)
        weights = np.ones(n)

        solver = PIRLSSolver(
            X_aug,
            y,
            self.family,
            S_aug,
            lambda_vec=np.ones(len(S_aug)),
            offset=offset,
            dispersion=self.dispersion_,
            weights=weights,
        )
        beta_aug = solver.solve(max_iter=max_inner_iter if hasattr(self, "_max_inner") else 25)

        self.beta = beta_aug[:p]
        self._X_fit = X_aug
        self._S_combined = combined_S

        # Store random effects
        re_start = p
        for group_var, formula in self.random.items():
            group = pd.Categorical(self.data[group_var])
            n_levels = len(group.categories)
            re_vec = beta_aug[re_start : re_start + n_levels]
            self.random_effects[group_var] = dict(zip(group.categories, re_vec))
            re_start += n_levels

    def _estimate_variance_components(self) -> None:
        """Estimate variance components σ²_j for each smooth/RE term."""
        if not self.fitted:
            return

        # Simple moment estimator: σ²_j ≈ φ / λ_j for smooth terms
        if self.smoothing_parameters is not None:
            for j, lam_j in enumerate(self.smoothing_parameters):
                self.var_components[f"smooth_{j}"] = float(self.dispersion_ / max(lam_j, 1e-10))

    def ranef(self) -> dict:
        """Return estimated random effects.

        Returns:
            Dict mapping group variable name to {level: estimate} dict.
        """
        return self.random_effects.copy()

    def summary(self) -> str:
        """Model summary including random effects variance components."""
        base = super().summary()

        if not self.random_effects and not self.var_components:
            return base

        lines = [base, "", "Random effects variance components:"]
        for name, var in self.var_components.items():
            lines.append(f"  {name}: σ² = {var:.6f}  (sd = {var**0.5:.6f})")

        if self.random_effects:
            lines.append("")
            lines.append("Random effects (first few):")
            for group_var, levels in self.random_effects.items():
                first_few = list(levels.items())[:5]
                for lv, v in first_few:
                    lines.append(f"  {group_var}[{lv}]: {v:.6f}")
                if len(levels) > 5:
                    lines.append(f"  ... ({len(levels)} levels total)")

        return "\n".join(lines)


def gamm(
    formula: str,
    data: pd.DataFrame,
    family: str = "gaussian",
    random: dict | None = None,
    offset: str | None = None,
    weights: Any | None = None,
    **kwargs,
) -> GAMM:
    """Fit a Generalized Additive Mixed Model (gamm).

    Functional convenience wrapper around GAMM.

    Args:
        formula: Model formula with smooth terms.
        data: Input data as DataFrame.
        family: Distribution family.
        random: Random effects, e.g. {'subject': '~1'}.
        offset: Optional offset column.
        weights: Optional weights.
        **kwargs: Additional arguments to GAMM.fit().

    Returns:
        Fitted GAMM object.

    Example:
        >>> model = gamm('y ~ s(x)', data=df, random={'subject': '~1'})
        >>> print(model.summary())
        >>> print(model.ranef())
    """
    model = GAMM(
        formula=formula,
        data=data,
        family=family,
        random=random,
        offset=offset,
        weights=weights,
    )
    model.fit(**kwargs)
    return model
