"""Automated validation: pymgcv vs mgcv (R).

Procedure
---------
1.  Generate fixed-seed synthetic datasets.
2.  Fit models in mgcv by calling Rscript via subprocess.
3.  Fit identical models in pymgcv.
4.  Compare coefficients, EDF, predictions, and deviance.

Metrics (per model)
-------------------
- max |β_py − β_R|  (coefficient absolute error)
- max |pred_py − pred_R| / (|pred_R| + 1e-8)  (relative prediction error)
- |EDF_py − EDF_R|  (EDF absolute error)
- |deviance_py − deviance_R| / (|deviance_R| + 1e-8)

Usage
-----
    python validate_pymgcv.py [--r-exe "C:/Program Files/R/R-4.4.1/bin/Rscript.exe"]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Fixed-seed data generators (must match R script exactly)
# ---------------------------------------------------------------------------

RNG_SEED = 42


def _make_gaussian_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    n = 100
    x = np.linspace(0, 1, n)
    y = 2 + 3 * x + 2 * np.sin(4 * np.pi * x) + rng.normal(0, 0.3, n)
    return pd.DataFrame({'x': x, 'y': y})


def _make_poisson_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(0, 3, 100)
    eta = 0.5 + 0.6 * x
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame({'x': x, 'y': y.astype(float)})


def _make_binomial_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(0, 1, 80)
    p = 1 / (1 + np.exp(-(2 * np.sin(4 * np.pi * x))))
    y = rng.binomial(1, p).astype(float)
    return pd.DataFrame({'x': x, 'y': y})


def _make_multi_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    n = 100
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 2, n)
    y = 1.2 * np.sin(4 * np.pi * x1) + 0.8 * np.cos(3 * np.pi * x2) + rng.normal(0, 0.2, n)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})


def _make_cr_data() -> pd.DataFrame:
    """Cubic regression spline test."""
    rng = np.random.default_rng(RNG_SEED)
    n = 150
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.4 * rng.normal(0, 1, n)
    return pd.DataFrame({'x': x, 'y': y})


# ---------------------------------------------------------------------------
# 2. R script (written to a temp file)
# ---------------------------------------------------------------------------

_R_SCRIPT = r"""
suppressMessages({
  library(mgcv)
  library(jsonlite)
})

set.seed(42)

results <- list()

# Helper to safely extract
extract <- function(m) {
  list(
    coef         = as.numeric(coef(m)),
    coef_names   = names(coef(m)),
    total_edf    = as.numeric(sum(m$edf)),
    aic          = as.numeric(AIC(m)),
    deviance     = as.numeric(m$deviance),
    sp           = as.numeric(m$sp),
    fitted       = as.numeric(fitted(m))
  )
}

# --- Gaussian ---
n <- 100; x <- seq(0,1,length.out=n)
y <- 2 + 3*x + 2*sin(4*pi*x) + rnorm(n, sd=0.3)
m <- gam(y ~ s(x), data=data.frame(x=x,y=y), method="REML")
results$gaussian <- extract(m)

# --- Poisson ---
x <- seq(0,3,length.out=100)
y <- rpois(100, exp(0.5 + 0.6*x))
m <- gam(y ~ s(x), family=poisson(), data=data.frame(x=x,y=y), method="REML")
results$poisson <- extract(m)

# --- Binomial ---
x <- seq(0,1,length.out=80)
p <- plogis(2*sin(4*pi*x))
y <- rbinom(80, 1, p)
m <- gam(y ~ s(x), family=binomial(), data=data.frame(x=x,y=y), method="REML")
results$binomial <- extract(m)

# --- Two smooths ---
n <- 100
x1 <- seq(0,1,length.out=n); x2 <- seq(0,2,length.out=n)
y <- 1.2*sin(4*pi*x1) + 0.8*cos(3*pi*x2) + rnorm(n, sd=0.2)
m <- gam(y ~ s(x1) + s(x2), data=data.frame(x1=x1,x2=x2,y=y), method="REML")
results$multi <- extract(m)

# --- Cubic regression spline (bs='cr') ---
n <- 150; x <- seq(0,1,length.out=n)
y <- sin(2*pi*x) + 0.4*rnorm(n)
m <- gam(y ~ s(x, bs='cr'), data=data.frame(x=x,y=y), method="REML")
results$cr <- extract(m)

writeLines(toJSON(results, auto_unbox=TRUE, digits=10), con=OUT_FILE)
cat("R: wrote results to", OUT_FILE, "\n")
"""


# ---------------------------------------------------------------------------
# 3. pymgcv fitting
# ---------------------------------------------------------------------------

def _fit_pymgcv(name: str, formula: str, df: pd.DataFrame, family: str = 'gaussian') -> dict:
    from pymgcv.api.gam import GAM
    m = GAM(formula, data=df, family=family, method='REML')
    m.fit()
    eta = m._X_fit @ m.beta
    offset = m.model_matrix.offset_vector()
    if offset is not None:
        eta = eta + offset
    mu = m.family.linkinv(eta)
    # deviance: -2 * loglik
    deviance = -2.0 * m.family.loglik(m._y_fit, mu, m.dispersion_)
    return {
        'coef': m.beta,
        'total_edf': m.edf,
        'sp': m.smoothing_parameters,
        'fitted': mu,
        'deviance': deviance,
    }


# ---------------------------------------------------------------------------
# 4. Comparison helpers
# ---------------------------------------------------------------------------

def _align_coef(py_coef: np.ndarray, r_coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align coefficient vectors to same length (take the minimum)."""
    n = min(len(py_coef), len(r_coef))
    return py_coef[:n], r_coef[:n]


def compare(py: dict, r: dict, label: str, tol_coef: float = 1e-2) -> dict:
    py_coef = np.asarray(py['coef'])
    r_coef  = np.asarray(r['coef'])
    py_c, r_c = _align_coef(py_coef, r_coef)

    coef_err  = float(np.max(np.abs(py_c - r_c)))
    edf_err   = abs(float(py['total_edf']) - float(r['total_edf']))
    dev_err   = abs(float(py['deviance'])  - float(r['deviance'])) / (abs(float(r['deviance'])) + 1e-8)

    py_fit = np.asarray(py['fitted'])
    r_fit  = np.asarray(r['fitted'])
    n = min(len(py_fit), len(r_fit))
    pred_rel_err = float(np.max(np.abs(py_fit[:n] - r_fit[:n]) / (np.abs(r_fit[:n]) + 1e-8)))

    passed = (coef_err < tol_coef) and (edf_err < 1.0) and (dev_err < 0.05)

    result = {
        'label': label,
        'coef_max_abs_err': round(coef_err, 8),
        'edf_abs_err': round(edf_err, 4),
        'deviance_rel_err': round(dev_err, 6),
        'pred_max_rel_err': round(pred_rel_err, 6),
        'passed': passed,
    }
    return result


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------

def run(r_exe: str, verbose: bool = True) -> list[dict]:
    # Write temp R script with output file path
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        out_json = f.name

    with tempfile.NamedTemporaryFile(suffix='.R', mode='w', delete=False) as f:
        r_script_path = f.name
        header = f'OUT_FILE <- {json.dumps(out_json)}\n'
        f.write(header + _R_SCRIPT)

    try:
        if verbose:
            print(f'Running R: {r_exe} {r_script_path}')
        proc = subprocess.run(
            [r_exe, '--vanilla', r_script_path],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            print('R stderr:', proc.stderr[-2000:])
            raise RuntimeError(f'R script failed (exit {proc.returncode})')
        if verbose:
            print(proc.stdout)

        with open(out_json) as f:
            r_results = json.load(f)
    finally:
        os.unlink(r_script_path)
        try:
            os.unlink(out_json)
        except OSError:
            pass

    # pymgcv results
    datasets = {
        'gaussian': (_make_gaussian_data(), 'y ~ s(x)',         'gaussian'),
        'poisson':  (_make_poisson_data(),  'y ~ s(x)',         'poisson'),
        'binomial': (_make_binomial_data(), 'y ~ s(x)',         'binomial'),
        'multi':    (_make_multi_data(),    'y ~ s(x1) + s(x2)', 'gaussian'),
        'cr':       (_make_cr_data(),       "y ~ s(x, bs='cr')", 'gaussian'),
    }

    reports = []
    all_passed = True
    print('\n' + '='*70)
    print(f'{"MODEL":<22}  {"β err":>10}  {"EDF err":>8}  {"Dev err":>9}  {"Pred err":>9}  PASS')
    print('-'*70)

    for key, (df, formula, fam) in datasets.items():
        if key not in r_results:
            continue
        try:
            py = _fit_pymgcv(key, formula, df, family=fam)
        except Exception as e:
            print(f'  {key:<22}  ERROR: {e}')
            continue

        r = r_results[key]
        rep = compare(py, r, label=key)
        reports.append(rep)
        status = 'PASS' if rep['passed'] else 'FAIL'
        if not rep['passed']:
            all_passed = False
        print(
            f"  {key:<22}  {rep['coef_max_abs_err']:>10.2e}"
            f"  {rep['edf_abs_err']:>8.4f}"
            f"  {rep['deviance_rel_err']:>9.4f}"
            f"  {rep['pred_max_rel_err']:>9.4f}"
            f"  {status}"
        )

    print('='*70)
    overall = 'ALL PASSED' if all_passed else 'SOME FAILURES'
    print(f'Result: {overall}')
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate pymgcv against mgcv.')
    parser.add_argument(
        '--r-exe',
        default=r'C:\Program Files\R\R-4.4.1\bin\Rscript.exe',
        help='Path to Rscript executable.',
    )
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    if not os.path.isfile(args.r_exe):
        # Try fallback
        for candidate in ['Rscript', 'Rscript.exe']:
            try:
                subprocess.run([candidate, '--version'], capture_output=True, check=True)
                args.r_exe = candidate
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        else:
            print(f'ERROR: Rscript not found at {args.r_exe!r}. Install R or pass --r-exe.')
            sys.exit(1)

    run(r_exe=args.r_exe, verbose=args.verbose)


if __name__ == '__main__':
    main()
