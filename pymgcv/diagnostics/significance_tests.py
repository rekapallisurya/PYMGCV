"""Smooth term significance tests.

Tests the null hypothesis H₀: f(x) = 0 for each smooth term.

For Gaussian family:
    - T-statistic: chisq = MSR / MSE (where MSR = model sum of squares)
    - Approximate degrees of freedom via EDF
    - F-statistic: F ≈ (Reduced Deviance - Full Deviance) / EDF_smooth / (dispersion)
    - p-value: P(χ² > chisq) or P(F > F_stat)

For non-Gaussian:
    - Approximate chi-square test using deviance difference
    - Dunn-Clagg, or Wald test approximations

References:
    - Wood, S. N. (2017): Generalized Additive Models, Ch. 4
    - Hastie, T. & Tibshirani, R. (1990): GLM extensions

Module exports:
    - SmoothTest: Test statistic class
    - test_smooth_terms: Functional API
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats


class SmoothTest:
    """Significance test for a smooth term.

    Attributes:
        smooth_name: Name of the smooth term.
        edf: Effective degrees of freedom.
        ref_df: Reference (unpenalized) degrees of freedom.
        test_stat: Test statistic value (F or chi-square).
        test_type: Type of test ('F', 'chisq').
        p_value: p-value from test.
    """

    def __init__(
        self,
        smooth_name: str,
        edf: float,
        ref_df: int,
        deviance_full: float,
        deviance_null: float,
        dispersion: float = 1.0,
        test_type: str = 'F',
    ) -> None:
        """Initialize smooth term test.

        Args:
            smooth_name: Name of smooth term.
            edf: Effective degrees of freedom of this smooth.
            ref_df: Reference (unpenalized) degrees of freedom.
            deviance_full: Deviance with this smooth included.
            deviance_null: Deviance without this smooth (null model).
            dispersion: Dispersion parameter φ.
            test_type: 'F' for F-test, 'chisq' for chi-square test.
        """
        self.smooth_name = smooth_name
        self.edf = float(edf)
        self.ref_df = int(ref_df)
        self.deviance_full = float(deviance_full)
        self.deviance_null = float(deviance_null)
        self.dispersion = float(dispersion)
        self.test_type = test_type

        self.test_stat = 0.0
        self.p_value = 1.0

        self._compute_test()

    def _compute_test(self) -> None:
        """Compute test statistic and p-value."""
        deviance_diff = self.deviance_null - self.deviance_full

        if self.test_type == 'F':
            # F-test
            if self.edf > 0 and self.dispersion > 0:
                self.test_stat = (deviance_diff / self.edf) / self.dispersion
                # Approximate F distribution: df1 ≈ edf, df2 ≈ large
                # Use chi-square approximation: edf * F ≈ χ²(edf)
                p = 1 - stats.f.cdf(self.test_stat, self.edf, 10000)
                self.p_value = max(0, min(1, p))  # Clamp to [0, 1]
            else:
                self.test_stat = np.nan
                self.p_value = 1.0

        elif self.test_type == 'chisq':
            # Chi-square test
            if self.dispersion > 0:
                self.test_stat = deviance_diff / self.dispersion
                p = 1 - stats.chi2.cdf(self.test_stat, df=self.edf)
                self.p_value = max(0, min(1, p))
            else:
                self.test_stat = np.nan
                self.p_value = 1.0

    def summary(self) -> str:
        """Return summary string."""
        return (
            f'{self.smooth_name:20s} '
            f'EDF={self.edf:6.2f}  '
            f'Ref.df={self.ref_df:3d}  '
            f'{self.test_type}={self.test_stat:8.4f}  '
            f'p={self.p_value:7.4f}'
        )


class SmoothTestSuite:
    """Run significance tests for all smooth terms.

    Attributes:
        tests: List of SmoothTest objects.
        overall_summary: Human-readable summary table.
    """

    def __init__(
        self,
        smooth_names: list[str],
        edf_smooth: dict[int, float],
        ref_dfs: list[int],
        deviance_full: float,
        deviance_null_dict: dict[int, float],
        dispersion: float = 1.0,
        test_type: str = 'F',
    ) -> None:
        """Initialize test suite.

        Args:
            smooth_names: List of smooth term names.
            edf_smooth: Dict of smooth_index → EDF.
            ref_dfs: List of reference DF per smooth.
            deviance_full: Deviance with all smooths included.
            deviance_null_dict: Dict of smooth_index → deviance without that smooth.
            dispersion: Dispersion parameter.
            test_type: 'F' or 'chisq'.
        """
        self.smooth_names = smooth_names
        self.edf_smooth = edf_smooth
        self.ref_dfs = ref_dfs
        self.deviance_full = deviance_full
        self.dispersion = dispersion
        self.test_type = test_type

        self.tests: list[SmoothTest] = []

        # Run tests
        for j, name in enumerate(smooth_names):
            edf = edf_smooth.get(j, np.nan)
            ref_df = ref_dfs[j] if j < len(ref_dfs) else 0
            dev_null = deviance_null_dict.get(j, np.nan)

            test = SmoothTest(
                smooth_name=name,
                edf=edf,
                ref_df=ref_df,
                deviance_full=deviance_full,
                deviance_null=dev_null,
                dispersion=dispersion,
                test_type=test_type,
            )
            self.tests.append(test)

    def summary(self) -> str:
        """Return summary table."""
        lines = [
            'Smooth term tests',
            '=================',
            f'Test type: {self.test_type}',
            '',
            f'{"Smooth Term":<20} {"EDF":>6} {"Ref.df":>8} {f"{self.test_type} Stat":>10} {"p-value":>8}',
            '-' * 60,
        ]
        for test in self.tests:
            lines.append(test.summary())
        return '\n'.join(lines)

    def p_values(self) -> list[float]:
        """Return p-values."""
        return [test.p_value for test in self.tests]

    def significant_terms(self, alpha: float = 0.05) -> list[str]:
        """Return names of significant terms at level α."""
        return [
            test.smooth_name for test in self.tests
            if test.p_value < alpha
        ]


def compute_smooth_tests(
    y_full: np.ndarray,
    y_residuals_full: np.ndarray,
    edf_smooth: dict[int, float],
    smooth_names: list[str],
    ref_dfs: list[int],
    dispersion: float = 1.0,
    null_deviances: Optional[dict[int, float]] = None,
) -> SmoothTestSuite:
    """Run smooth term tests.

    Args:
        y_full: Full response vector.
        y_residuals_full: Residuals from full model.
        edf_smooth: EDF per smooth term.
        smooth_names: Names of smooth terms.
        ref_dfs: Reference DF per smooth.
        dispersion: Dispersion parameter.
        null_deviances: Dict of smooth_index → deviance when that smooth omitted.

    Returns:
        SmoothTestSuite object with results.
    """
    # Full deviance
    deviance_full = np.sum(y_residuals_full**2)

    if null_deviances is None:
        # Use approximate null deviances (TODO: implement proper deletion)
        null_deviances = {j: deviance_full for j in range(len(smooth_names))}

    suite = SmoothTestSuite(
        smooth_names=smooth_names,
        edf_smooth=edf_smooth,
        ref_dfs=ref_dfs,
        deviance_full=deviance_full,
        deviance_null_dict=null_deviances,
        dispersion=dispersion,
        test_type='F',
    )

    return suite


# Backward-compatible alias
test_smooth_terms = compute_smooth_tests
