# pymgcv

**Production-grade Generalized Additive Models with numerical equivalence to R's mgcv.**

[![CI](https://github.com/surya/pymgcv/actions/workflows/ci.yml/badge.svg)](https://github.com/surya/pymgcv/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/pymgcv.svg)](https://pypi.org/project/pymgcv/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is pymgcv?

`pymgcv` is a complete Python implementation of Generalized Additive Models (GAMs)
that achieves **numerical equivalence** with Simon Wood's R package
[mgcv](https://cran.r-project.org/web/packages/mgcv/) within tolerance 1e-6.

## Key Features

- **Full mgcv parity** — coefficients, EDF, p-values, predictions, AIC/GCV/REML match R
- **13+ smooth types** — thin plate, cubic, B-spline, P-spline, cyclic, tensor products, random effects, …
- **All major families** — Gaussian, Poisson, Binomial, Gamma, Tweedie (with power estimation), …
- **Production-grade** — fully typed, 231 tests passing, comprehensive diagnostics
- **GPU acceleration** — optional JAX backend for large datasets
- **Native C/Fortran** — compiled LAPACK kernels with pure-Python fallback

## Quick Install

```bash
pip install pymgcv
```

## Quick Example

```python
from pymgcv import GAM
import pandas as pd

model = GAM("y ~ s(x1) + s(x2)", data=df)
model.fit()
print(model.summary())
```

Explore the [Quickstart guide](getting-started/quickstart.md) for more.
