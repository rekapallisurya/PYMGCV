<p align="center">
  <h1 align="center">pymgcv</h1>
  <p align="center">
    <strong>Production-Grade Generalized Additive Models for Python</strong>
  </p>
  <p align="center">
    <a href="https://github.com/rekapallisurya/PYMGCV/actions"><img src="https://github.com/rekapallisurya/PYMGCV/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://pypi.org/project/pymgcv/"><img src="https://img.shields.io/pypi/v/pymgcv.svg" alt="PyPI"></a>
    <a href="https://pypi.org/project/pymgcv/"><img src="https://img.shields.io/pypi/pyversions/pymgcv.svg" alt="Python"></a>
    <a href="https://github.com/rekapallisurya/PYMGCV/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
    <a href="https://codecov.io/gh/rekapallisurya/PYMGCV"><img src="https://codecov.io/gh/rekapallisurya/PYMGCV/branch/main/graph/badge.svg" alt="Coverage"></a>
  </p>
</p>

---

**pymgcv** is a complete Python implementation of Generalized Additive Models (GAMs) achieving **numerical equivalence with R's mgcv** (Simon Wood) within 1%. It is designed for data scientists, actuaries, and researchers who need production-grade nonlinear modelling without switching to R.

```python
from pymgcv import GAM, Tweedie

model = GAM(
    formula="spend ~ s(age) + s(income_k) + s(contacts)",
    family=Tweedie(estimate_power=True),
    method="REML",
)
model.fit(df)
print(model.summary())
```

---

## Table of Contents

- [Why pymgcv?](#why-pymgcv)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Distributions](#supported-distributions)
- [Smooth Basis Types](#smooth-basis-types)
- [API Reference](#api-reference)
- [R mgcv Equivalence](#r-mgcv-equivalence)
- [Examples](#examples)
- [Architecture](#architecture)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Why pymgcv?

| | pymgcv | pyGAM | statsmodels GAM |
|---|---|---|---|
| R mgcv parity | **< 1% Δ** | No | No |
| Tweedie (est. power) | ✅ | ❌ | ❌ |
| Tensor products (te/ti) | ✅ | ❌ | ❌ |
| REML smoothing selection | ✅ | ❌ | ❌ |
| Formula interface | R-style | Grid search | statsmodels |
| GPU acceleration (JAX) | ✅ | ❌ | ❌ |
| Basis types | 12+ | 1 | 2 |
| Production tested | 246 tests | — | — |

---

## Features

### Core Engine
- **PIRLS Solver** — Penalized Iteratively Reweighted Least Squares with step-halving and warm-start convergence
- **MAGIC Optimizer** — Newton optimization of smoothing parameters with backtracking line search
- **REML / GCV / UBRE / ML** — Four smoothing parameter selection criteria
- **EDF Computation** — Effective degrees of freedom via `tr((X'WX + Sλ)⁻¹ X'WX)`

### Smoothing
- **12 basis types** — Thin plate regression splines, cubic regression/shrinkage, B-splines, P-splines, cyclic splines, tensor products, random effects, Gaussian process, adaptive, factor smooth interactions
- **Penalty matrices** — Automatic construction with QR constraint absorption and Demmler–Reinsch orthogonalization
- **Tensor products** — `te()`, `ti()`, `t2()` with Kronecker penalty structures
- **Shrinkage selection** — `select=True` adds an extra penalty for automatic variable selection

### Distributions
- **9 families** — Gaussian, Poisson, Binomial, Gamma, Tweedie, Negative Binomial, Inverse Gaussian, Beta, Gaulss (location-scale)
- **Tweedie power estimation** — Automatic *p* search via Laplace REML + Wright function (mirrors R's `tw()`)
- **Link functions** — identity, log, logit, inverse, with derivatives for Newton updates

### Diagnostics & Output
- **R-style `summary()`** — Parametric coefficients table + approximate smooth significance tests (Wood 2013)
- **`gam.check()`** — Residual diagnostics + k-adequacy test
- **Concurvity** — Pairwise and overall smooth collinearity diagnostics
- **Influence** — Leverage, Cook's distance, DFBETAS
- **Visualization** — 1D/2D smooth plots, 3D tensor surfaces, diagnostic panels

### Performance
- **Native C extension** — Compiled LAPACK solver (`dposv`) for SPD systems
- **JAX GPU backend** — Optional auto-differentiation and GPU-accelerated matrix operations
- **Fortran LAPACK** — via SciPy wrappers with NumPy fallback

---

## Installation

### From PyPI (coming soon)

```bash
pip install pymgcv
```

### From Source

```bash
git clone https://github.com/rekapallisurya/PYMGCV.git
cd PYMGCV
pip install -e ".[dev]"
```

### Optional Extras

```bash
pip install pymgcv[gpu]       # JAX GPU acceleration
pip install pymgcv[viz]       # matplotlib + plotly visualization
pip install pymgcv[full]      # everything
pip install pymgcv[dev]       # development (testing, linting, docs)
```

### Requirements

| Dependency | Version | Required |
|-----------|---------|----------|
| Python | ≥ 3.11 | Yes |
| numpy | ≥ 1.21 | Yes |
| scipy | ≥ 1.7 | Yes |
| pandas | ≥ 1.3 | Yes |
| matplotlib | ≥ 3.5 | Optional (viz) |
| jax + jaxlib | latest | Optional (GPU) |

### Native Backend

pymgcv includes an optional C extension for LAPACK-accelerated solves. If a C compiler is unavailable, installation succeeds with a NumPy/SciPy fallback:

```python
from pymgcv.linalg import backend_info
print(backend_info())
# {'native_c': True, 'scipy_lapack': True, 'numpy_fallback': True}
```

---

## Quick Start

### 1. Gaussian GAM

```python
from pymgcv import GAM
import pandas as pd
import numpy as np

np.random.seed(42)
n = 500
x = np.linspace(0, 2 * np.pi, n)
y = np.sin(x) + 0.3 * np.random.randn(n)
df = pd.DataFrame({"x": x, "y": y})

model = GAM("y ~ s(x)", data=df, method="REML")
model.fit()
print(model.summary())
```

### 2. Tweedie GAM (Insurance Pricing)

```python
from pymgcv import GAM, Tweedie

model = GAM(
    formula=(
        "capped_claims_kusd ~ offset(log_duration) + "
        "s(vehicle_age, k=8) + s(driver_age, k=10) + "
        "s(bonus_malus, k=8) + class_B + class_C + class_D"
    ),
    data=df,
    family=Tweedie(power=1.5),
    method="REML",
    gamma=1.2,
)
model.fit()

# Predict on response scale
df["predicted"] = model.predict(df, scale="response")
```

### 3. Campaign Spend with Estimated Power

```python
from pymgcv import GAM, Tweedie

model = GAM(
    formula="spend ~ s(age) + s(income_k) + s(contacts)",
    family=Tweedie(estimate_power=True),  # mirrors R's tw()
    method="REML",
)
model.fit(df)

# The estimated power parameter
print(f"Tweedie power: {model.family.power:.3f}")
```

### 4. Tensor Product Smooths

```python
model = GAM(
    "y ~ te(x1, x2) + s(x3)",
    data=df,
    method="REML",
)
model.fit()
```

### 5. Visualization

```python
from pymgcv import plot_smooth, plot_residuals
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, var in enumerate(["age", "income_k", "contacts"]):
    plot_smooth(model, var_name=var, ax=axes[i])
plt.tight_layout()
plt.show()
```

### 6. Model Comparison

```python
from pymgcv import aic, bic

model_1 = GAM("y ~ s(x1) + s(x2)", data=df).fit()
model_2 = GAM("y ~ s(x1) + s(x2) + s(x3)", data=df).fit()

print(f"Model 1 AIC: {aic(model_1):.1f}")
print(f"Model 2 AIC: {aic(model_2):.1f}")
```

---

## Supported Distributions

| Family | Link | Variance V(μ) | Use Case |
|--------|------|---------------|----------|
| `'gaussian'` | identity | σ² | Continuous response |
| `'poisson'` | log | μ | Count data |
| `'binomial'` | logit | μ(1−μ) | Binary / proportion |
| `'gamma'` | log | μ² | Positive continuous |
| `Tweedie(p=1.5)` | log | φμᵖ | Insurance, zero-inflated |
| `Tweedie(estimate_power=True)` | log | φμᵖ (p estimated) | Auto power selection |
| `'negbinomial'` | log | μ + μ²/θ | Over-dispersed counts |
| `'inverse_gaussian'` | 1/μ² | μ³ | Positive, right-skewed |
| `'beta'` | logit | μ(1−μ)/(1+φ) | Proportions on (0,1) |

---

## Smooth Basis Types

| Code | Full Name | Penalty | Typical Use |
|------|-----------|---------|-------------|
| `tp` | Thin plate regression spline | Identity (after reparameterization) | **Default**, general-purpose |
| `cr` | Cubic regression spline | Natural spline penalty | Evenly-spaced data |
| `cs` | Cubic shrinkage spline | Extra shrinkage penalty | Variable selection |
| `bs` | B-spline | Difference penalty | Flexible, local control |
| `ps` | P-spline (Eilers–Marx) | Difference penalty | Equally-spaced basis |
| `cc` | Cyclic cubic spline | Wrap-around constraint | Seasonal / circular data |
| `cp` | Cyclic P-spline | Wrap-around + difference | Periodic smooth |
| `re` | Random effect | Ridge (identity) | Random intercepts/slopes |
| `gp` | Gaussian process smooth | Matérn covariance | Spatial data |
| `ad` | Adaptive smooth | Locally varying penalty | Heterogeneous smoothness |
| `fs` | Factor smooth interaction | Per-level smooth | Group-specific curves |
| `sz` | Factor deviation smooth | Deviation from mean | Difference from baseline |

**Formula syntax:**
```python
"y ~ s(x1) + s(x2, bs='cr', k=15) + te(x1, x2) + s(x3, by=group) + re(subject)"
```

---

## API Reference

### Core Classes

```python
from pymgcv import GAM, BAM, GAMM

# GAM — Generalized Additive Model
model = GAM(
    formula: str,              # R-style formula
    data: pd.DataFrame = None, # training data (or pass to .fit())
    family: str | Family = 'gaussian',
    method: str = 'REML',      # 'REML', 'GCV', 'ML', 'UBRE'
    gamma: float = 1.0,        # smoothing inflation factor
    sp: array = None,          # fixed smoothing parameters
    select: bool = False,      # shrinkage variable selection
    control: dict = None,      # {'maxit': 50, 'epsilon': 1e-4}
    weights_col: str = None,   # observation weights column
    knots: dict = None,        # per-variable knot positions
)
model.fit(data=None, verbose=False)

# BAM — Fast approximation for large n
model = BAM(formula, data, family, ...)

# GAMM — GAM with random effects
model = GAMM(formula, data, family, random={'subject': '~1'}, ...)
```

### Prediction & Inference

```python
# Predictions
y_hat = model.predict(newdata, scale='response')  # or 'link'

# Confidence intervals
ci = model.confidence_intervals(newdata, ci=0.95)

# Standard errors
se = model.standard_errors(newdata)

# Summary (R mgcv-style)
print(model.summary())
```

### Diagnostics

```python
from pymgcv import gam_check, aic, bic, anova_gam

# Basis adequacy + residual diagnostics
gam_check(model)

# k-index check
model.k_check()

# Model comparison
anova_gam(model_1, model_2)
print(f"AIC: {aic(model)}, BIC: {bic(model)}")
```

### Visualization

```python
from pymgcv import plot_smooth, plot_residuals

# Individual smooth plots with CIs
plot_smooth(model, var_name='age', ci=0.95)

# Diagnostic panel
plot_residuals(model)
```

### Convenience Functions

```python
from pymgcv import s, te, ti, re

# Programmatic formula building
formula = "y ~ " + s("x1", k=10) + " + " + te("x1", "x2") + " + " + re("subject")
```

---

## R mgcv Equivalence

pymgcv is validated against R's mgcv on identical datasets. Key comparison:

### Campaign Model (Tweedie, estimated power)

| Metric | pymgcv | R mgcv | Δ |
|--------|--------|--------|---|
| Tweedie power (*p*) | 1.331 | 1.344 | 1.0% |
| Dispersion (*φ*) | 5.46 | 5.91 | 7.6% |
| Intercept | −1.255 | −1.255 | ~0% |
| s(age) EDF | 1.000 | 1.009 | 0.9% |
| s(income_k) EDF | 1.000 | 1.001 | ~0% |
| s(contacts) EDF | 1.007 | 1.006 | 0.1% |
| Deviance explained | 13.68% | 13.6% | 0.6% |

### Insurance Loss Cost (Tweedie p=1.5, fixed)

With fixed power, **all outputs match within < 0.5%** (floating-point tolerance).

![Campaign Comparison](docs/campaign_comparison.png)
![Loss Cost Comparison](docs/losscost_comparison.png)

### Why Small Differences Exist

```
Tweedie power estimation        ← Laplace approx (pymgcv) vs exact REML (R Fortran)
         ↓
Dispersion estimation            ← φ = Pearson/df changes with p
         ↓
F-statistics (8-9% when p differs)
         ↓
All other metrics agree within 1%
```

When *p* is fixed, differences reduce to sub-0.5% (Cholesky/knot placement only).

See [EXAMPLES.md](EXAMPLES.md) and [RMGCV_VS_PYMGCV_COMPARISON.md](RMGCV_VS_PYMGCV_COMPARISON.md) for full details.

---

## Examples

### Included Examples

| File | Description |
|------|-------------|
| [Campaign.py](Campaign.py) | Tweedie GAM with estimated power — marketing spend prediction |
| [loss_cost_model.py](loss_cost_model.py) | Insurance loss-cost model with offset, gamma=1.2, Excel export |
| [examples/MY_FIRST_GAM.py](examples/MY_FIRST_GAM.py) | Beginner tutorial |
| [examples/insurance_pricing_demo.py](examples/insurance_pricing_demo.py) | Full actuarial workflow |
| [examples/comprehensive_family_examples.py](examples/comprehensive_family_examples.py) | All 9 distribution families |
| [examples/comparison_with_R.py](examples/comparison_with_R.py) | Side-by-side R mgcv comparison |
| [EXAMPLES.md](EXAMPLES.md) | Worked examples with R comparison plots |

### R Equivalent

Every pymgcv model has a direct R translation:

```r
# R mgcv
library(mgcv)
model <- gam(spend ~ s(age) + s(income_k) + s(contacts),
             data = df, family = tw(), method = "REML")
summary(model)
```

```python
# pymgcv (identical API)
from pymgcv import GAM, Tweedie
model = GAM("spend ~ s(age) + s(income_k) + s(contacts)",
            family=Tweedie(estimate_power=True), method="REML")
model.fit(df)
print(model.summary())
```

---

## Architecture

```
pymgcv/
├── api/                    # User-facing API
│   ├── gam.py              #   GAM class (fit + predict)
│   ├── bam.py              #   BAM (large n via QR reduction)
│   ├── gamm.py             #   GAMM (mixed effects)
│   ├── predict.py          #   Predictor with CI/SE
│   ├── summary.py          #   R-style model summary
│   ├── plot.py             #   Smooth + diagnostic plots
│   ├── gam_check.py        #   gam.check() + k-adequacy
│   ├── gam_auto.py         #   Automatic variable selection
│   └── model_comparison.py #   anova_gam, AIC, BIC
├── smooth/                 # Basis function generators
│   ├── thin_plate.py       #   TPRS (eigen-reparameterized)
│   ├── cubic_spline.py     #   Cubic regression / shrinkage
│   ├── bspline.py          #   B-spline basis
│   ├── pspline.py          #   P-spline (Eilers-Marx)
│   ├── cyclic_spline.py    #   Cyclic cubic / P-spline
│   ├── random_effect.py    #   Random effect (ridge)
│   ├── tensor_product.py   #   te(), ti(), t2()
│   └── advanced.py         #   GP, adaptive, factor smooth
├── distributions/          # Exponential family
│   ├── family_base.py      #   9 families + link functions
│   └── tweedie.py          #   Power search + Wright function
├── optimizer/              # Fitting engines
│   ├── pirls.py            #   PIRLS inner loop
│   ├── magic_optimizer.py  #   MAGIC outer Newton loop
│   ├── reml_objective.py   #   REML criterion + gradient
│   ├── gcv.py              #   GCV criterion
│   ├── edf.py              #   Effective degrees of freedom
│   └── jax_acceleration.py #   Optional JAX GPU backend
├── penalties/              # Penalty matrix construction
│   ├── penalty_matrix.py   #   S matrices + block assembly
│   ├── demmler_reinsch.py  #   QR constraint absorption
│   └── smoothing_param.py  #   Lambda management
├── linalg/                 # Linear algebra backends
│   ├── cholesky_solver.py  #   Cholesky SPD solver
│   ├── penalized_solver.py #   (X'WX + Sλ)β = X'Wz
│   ├── native_engine.py    #   C extension + Fortran LAPACK
│   └── sparse_utils.py     #   Block matrix utilities
├── diagnostics/            # Model diagnostics
│   ├── residuals.py        #   Deviance, Pearson, working
│   ├── significance_tests.py # Smooth F-tests (Wood 2013)
│   ├── concurvity.py       #   Smooth collinearity
│   └── influence.py        #   Leverage, Cook's D, DFBETAS
├── visualization/          # Plotting
│   ├── smooth_plots.py     #   1D/2D smooth effect plots
│   ├── surface_plots.py    #   3D tensor surfaces
│   └── diagnostics_plots.py #  Residual diagnostic panels
├── utils/                  # Utilities
│   ├── formula_parser.py   #   R-style formula parsing
│   └── model_matrix.py     #   Design matrix construction
└── config.py               # Default hyperparameters
```

---

## Testing

```bash
# Run full test suite (246 tests)
pytest

# With coverage
pytest --cov=pymgcv

# R mgcv equivalence tests only
pytest tests/test_validation_mgcv.py tests/test_thin_plate_mgcv_equivalence.py -v
```

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/rekapallisurya/PYMGCV.git
cd PYMGCV
pip install -e ".[dev]"
pre-commit install
pytest
```

---

## Citation

```bibtex
@software{pymgcv2026,
  title  = {pymgcv: Production-Grade Generalized Additive Models for Python},
  author = {Surya Rekapalli},
  year   = {2026},
  url    = {https://github.com/rekapallisurya/PYMGCV}
}
```

---

## References

- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC Press.
- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *JRSS-B*, 73(1), 3–36.
- Wood, S. N. (2013). On p-values for smooth components of an extended generalized additive model. *Biometrika*, 100(1), 221–228.
- Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in Sobolev spaces.
- Eilers, P. H. C. & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89–121.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Numerical validation and design inspiration from Simon Wood's R package **mgcv**.
