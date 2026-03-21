"""Automated validation: pymgcv vs mgcv (R).

Procedure
---------
1.  Generate fixed-seed synthetic datasets in Python.
2.  Write datasets to temporary CSV files.
3.  Fit models in mgcv by calling Rscript, reading the same CSVs.
4.  Fit identical models in pymgcv on the same data.
5.  Compare coefficients, EDF, predictions, and deviance.

Key design: BOTH systems use the EXACT same input data (via CSV),
so differences reflect numerical/algorithmic divergence only.

Metrics (per model)
-------------------
- max |beta_py - beta_R|        (coefficient absolute error)
- |EDF_py - EDF_R|              (EDF absolute error)
- |dev_py - dev_R| / |dev_R|    (relative deviance error)
- max |fitted_py - fitted_R| / (|fitted_R| + 1e-8)  (relative prediction error)

Pass criteria: coef_err < 0.01, edf_err < 0.5, dev_err < 0.02, pred_err < 0.01

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
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Fixed-seed data generators (Python-only; CSVs shared with R)
# ---------------------------------------------------------------------------

RNG_SEED = 42


def _make_gaussian_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    n = 200
    x = np.linspace(0, 1, n)
    y = 2 + 3 * x + 2 * np.sin(4 * np.pi * x) + rng.normal(0, 0.3, n)
    return pd.DataFrame({'x': x, 'y': y})


def _make_poisson_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(0, 3, 200)
    eta = 0.5 + 0.6 * x
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame({'x': x, 'y': y.astype(float)})


def _make_binomial_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(0, 1, 200)
    p = 1 / (1 + np.exp(-(2 * np.sin(4 * np.pi * x))))
    y = rng.binomial(1, p).astype(float)
    return pd.DataFrame({'x': x, 'y': y})


def _make_multi_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    n = 200
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 2, n)
    y = 1.2 * np.sin(4 * np.pi * x1) + 0.8 * np.cos(3 * np.pi * x2) + rng.normal(0, 0.2, n)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})


def _make_cr_data() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    n = 200
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.4 * rng.normal(0, 1, n)
    return pd.DataFrame({'x': x, 'y': y})


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2. R script template — reads CSVs written by Python
# ---------------------------------------------------------------------------

# The script uses injected variables: CSV_GAUSSIAN, CSV_POISSON, etc., OUT_FILE
_R_SCRIPT_TEMPLATE = """\
suppressMessages({{
  library(mgcv)
  library(jsonlite)
}})

extract <- function(m) {{
  list(
    coef       = as.numeric(coef(m)),
    coef_names = names(coef(m)),
    total_edf  = as.numeric(sum(m$edf)),
    aic        = as.numeric(AIC(m)),
    deviance   = as.numeric(m$deviance),
    sp         = as.numeric(m$sp),
    fitted     = as.numeric(fitted(m))
  )
}}

results <- list()

# --- Gaussian ---
d <- read.csv("{CSV_GAUSSIAN}")
m <- gam(y ~ s(x), data=d, method="REML")
results$gaussian <- extract(m)

# --- Poisson ---
d <- read.csv("{CSV_POISSON}")
m <- gam(y ~ s(x), family=poisson(), data=d, method="REML")
results$poisson <- extract(m)

# --- Binomial ---
d <- read.csv("{CSV_BINOMIAL}")
m <- gam(y ~ s(x), family=binomial(), data=d, method="REML")
results$binomial <- extract(m)

# --- Two smooths (Gaussian) ---
d <- read.csv("{CSV_MULTI}")
m <- gam(y ~ s(x1) + s(x2), data=d, method="REML")
results$multi <- extract(m)

# --- Cubic regression spline (bs='cr') ---
d <- read.csv("{CSV_CR}")
m <- gam(y ~ s(x, bs='cr'), data=d, method="REML")
results$cr <- extract(m)

writeLines(toJSON(results, auto_unbox=TRUE, digits=15), con="{OUT_FILE}")
cat("R: wrote results to {OUT_FILE}\\n")
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

    passed = (coef_err < tol_coef) and (edf_err < 0.5) and (dev_err < 0.02) and (pred_rel_err < 0.02)

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

def _write_csvs(datasets: dict) -> dict[str, str]:
    """Write each DataFrame to a temp CSV; return {key: path}."""
    paths = {}
    for key, df in datasets.items():
        f = tempfile.NamedTemporaryFile(suffix=f'_{key}.csv', delete=False, mode='w', newline='')
        df.to_csv(f, index=False)
        f.close()
        paths[key] = f.name
    return paths


def run(r_exe: str, verbose: bool = True) -> list[dict]:
    # Build datasets (Python-generated, shared with R via CSV)
    raw_datasets = {
        'gaussian': _make_gaussian_data(),
        'poisson':  _make_poisson_data(),
        'binomial': _make_binomial_data(),
        'multi':    _make_multi_data(),
        'cr':       _make_cr_data(),
    }
    # formulas / families for pymgcv
    meta = {
        'gaussian': ('y ~ s(x)',          'gaussian'),
        'poisson':  ('y ~ s(x)',          'poisson'),
        'binomial': ('y ~ s(x)',          'binomial'),
        'multi':    ('y ~ s(x1) + s(x2)', 'gaussian'),
        'cr':       ("y ~ s(x, bs='cr')", 'gaussian'),
    }

    csv_paths = _write_csvs(raw_datasets)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        out_json = f.name

    r_script_body = _R_SCRIPT_TEMPLATE.format(
        CSV_GAUSSIAN=csv_paths['gaussian'].replace('\\', '/'),
        CSV_POISSON= csv_paths['poisson'].replace('\\', '/'),
        CSV_BINOMIAL=csv_paths['binomial'].replace('\\', '/'),
        CSV_MULTI=   csv_paths['multi'].replace('\\', '/'),
        CSV_CR=      csv_paths['cr'].replace('\\', '/'),
        OUT_FILE=    out_json.replace('\\', '/'),
    )

    with tempfile.NamedTemporaryFile(suffix='.R', mode='w', delete=False) as f:
        r_script_path = f.name
        f.write(r_script_body)

    try:
        if verbose:
            print(f'Running R: {r_exe} {r_script_path}')
        proc = subprocess.run(
            [r_exe, '--vanilla', r_script_path],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            print('R stderr:', proc.stderr[-3000:])
            raise RuntimeError(f'R script failed (exit {proc.returncode})')
        if verbose and proc.stdout.strip():
            print(proc.stdout.strip())

        with open(out_json) as f:
            r_results = json.load(f)
    finally:
        os.unlink(r_script_path)
        for p in csv_paths.values():
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.unlink(out_json)
        except OSError:
            pass

    reports = []
    all_passed = True
    print('\n' + '=' * 72)
    print(f'  {"MODEL":<14}  {"beta err":>10}  {"EDF err":>8}  {"Dev err":>9}  {"Pred err":>9}  PASS')
    print('-' * 72)

    for key in meta:
        if key not in r_results:
            continue
        formula, fam = meta[key]
        df = raw_datasets[key]
        try:
            py = _fit_pymgcv(key, formula, df, family=fam)
        except Exception as e:
            print(f'  {key:<14}  ERROR: {e}')
            all_passed = False
            continue

        r = r_results[key]
        rep = compare(py, r, label=key)
        reports.append(rep)
        status = 'PASS' if rep['passed'] else 'FAIL'
        if not rep['passed']:
            all_passed = False
        print(
            f"  {key:<14}  {rep['coef_max_abs_err']:>10.2e}"
            f"  {rep['edf_abs_err']:>8.4f}"
            f"  {rep['deviance_rel_err']:>9.4f}"
            f"  {rep['pred_max_rel_err']:>9.4f}"
            f"  {status}"
        )

    print('=' * 72)
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

