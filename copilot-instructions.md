---
description: "Workspace-wide guidance for pymgcv package development—numerical equivalence with mgcv (Simon Wood) within tolerance 1e-6. All code is production-grade, fully typed, with mathematical documentation."
---

# pymgcv: Production-Grade GAM Package

## Vision

Implement a **complete, numerical equivalent** of R's `mgcv` package in Python 3.11+.

**Target Tolerance**: Coefficients, EDF, predictions, p-values, AIC/GCV/REML must match within **1e-6**.

**Scope**: 21 implementation steps across 4 phases.
- **Phase 1**: Foundations (Steps 1–5)
- **Phase 2**: Core Solver (Steps 6–11)
- **Phase 3**: Extensions (Steps 12–15)
- **Phase 4**: Output & Validation (Steps 16–21)

## Standards

### Code Quality
- **No pseudocode**: All modules are production-ready, fully typed.
- **Type hints everywhere**: Python 3.11+ with `from __future__ import annotations`.
- **Mathematical documentation**: Every algorithm includes inline docstrings with KaTeX formulas.
- **Docstring format**: Numpy-style with Parameters, Returns, Notes (mathematical context), Examples, References.

### Numerical Stability
- Use JAX for autodiff and GPU ops.
- Use scipy for linear algebra (Cholesky, QR, eigendecomposition).
- Apply Demmler–Reinsch orthogonalization to stabilize penalty matrices.
- Validate against mgcv within 1e-6 *before* proceeding to next step.

### Testing
- Unit tests for each component (basis, penalty, solver, optimizer).
- Integration tests comparing `pymgcv` output to `mgcv` on standard datasets.
- Test file: `tests/mgcv_comparison_tests.py`.
- Fixture data: R datasets (mtcars, AirPassengers, etc.) exported as CSV/pickle.

### Package Structure
```
pymgcv/
├── __init__.py              # API entry points
├── api/                     # High-level user-facing APIs
│   ├── gam.py              # Main GAM class
│   ├── bam.py              # Big additive models (future)
│   ├── gam_auto.py         # Automatic variable selection
│   ├── predict.py          # Prediction interface
│   ├── summary.py          # Model summaries
│   └── plot.py             # Visualization
├── smooth/                 # Basis & smooth term implementations
│   ├── smooth_basis.py     # Abstract base, registrar
│   ├── thin_plate.py       # TPRS basis
│   ├── cubic_spline.py     # Cubic spline basis
│   ├── tensor_product.py   # Tensor products (te, ti)
│   └── random_effect.py    # Random effects (re)
├── penalties/              # Penalty matrices
│   ├── penalty_matrix.py   # Base penalty construction
│   ├── smoothing_param.py  # λ management, MAGIC optimizer
│   └── demmler_reinsch.py  # Orthogonalization
├── optimizer/              # Solvers & optimization
│   ├── pirls.py            # Penalized IRLS
│   ├── magic_optimizer.py  # MAGIC (outer loop for λ)
│   └── reml_objective.py   # REML score, gradients, Hessian
├── linalg/                 # Linear algebra utilities
│   ├── cholesky_solver.py  # Cholesky factorization
│   ├── qr_decomposition.py # QR solver
│   ├── eigen_decomposition.py  # Symmetric eigen
│   └── trace_utils.py      # trace(A⁻¹ B) via CG, etc.
├── distributions/          # Exponential family GLM families
│   ├── family_base.py      # Abstract family
│   ├── gaussian.py         # Gaussian
│   ├── poisson.py          # Poisson
│   ├── gamma.py            # Gamma
│   └── tweedie.py          # Tweedie (variance power p)
├── diagnostics/            # Model diagnostics
│   ├── residuals.py        # Deviance, Pearson, etc.
│   ├── influence.py        # Hat matrix, leverage
│   └── concurvity.py       # Concurvity index
├── visualization/         # Plotting
│   ├── smooth_plots.py     # 1D/2D smooth effects
│   ├── surface_plots.py    # 3D tensor products
│   └── diagnostics_plots.py  # Residual plots, QQ, etc.
├── utils/                  # Utilities
│   ├── formula_parser.py   # Parse "y ~ s(x) + te(u,v)"
│   ├── model_matrix.py     # X assembly
│   ├── constraint_matrix.py  # Linear constraints
│   └── data_utils.py       # Data normalization, etc.
└── tests/
    ├── mgcv_comparison_tests.py
    ├── fixtures/           # CSV/pickle R datasets
    └── (unit test modules)
```

### Phase Workflow

**Before Starting Any Phase:**
1. Review this file to confirm standards.
2. Understand dependencies: Does current phase depend on prior steps?
3. Design classes/functions with full type hints and docstrings.

**During Implementation:**
1. Write production-quality code (no TODOs or stubs).
2. Include inline algorithm documentation (refs to Wood's papers, standard references).
3. Create unit tests as you code.
4. Validate outputs against mgcv (use R comparison script).

**After Completing Phase:**
1. Run full test suite.
2. Ensure all files have docstrings, type hints, references.
3. Update CHANGELOG with phase summary.
4. Document any design decisions in architectural notes.

## Key Mathematical References

- **Wood, S. N. (2017)**: "Generalized Additive Models: An Introduction with R" (2nd ed.) — the bible.
- **Thin Plate Splines**: Duchon (1977), Wood (2003) on R basis.
- **Penalty Matrices**: Derived from Green & Silverman (1994), implemented via Demmler–Reinsch.
- **PIRLS**: Iteratively reweighted least squares for GLM; standard in McCullagh & Nelder (1989).
- **MAGIC**: Wood & Farouki (1996) on smoothing parameter optimization; implemented as nested loops in mgcv.
- **REML**: Restricted maximum likelihood; Wood (2011) on GAM smoothing parameter selection via REML.
- **Tweedie**: Dunn & Smyth (2005) on Tweedie distributions; Hastie & Tibshirani (1990) on GLM dispersion.

## Environment & Dependencies

**Python**: 3.11+

**Core**:
- numpy ≥ 1.21
- scipy ≥ 1.7  
- pandas ≥ 1.3

**Autodiff & GPU**:
- jax ≥ 0.3.0
- jaxlib ≥ 0.3.0

**Visualization**:
- matplotlib ≥ 3.4
- plotly ≥ 5.0

**Development**:
- pytest ≥ 7.0
- pytest-cov
- black, isort, pylint
- mypy (strict mode)

## Validation Strategy

Each step is validated via:

1. **Unit tests**: Core functionality (basis, penalty, solver).
2. **Numerical regression tests**: Compare outputs to mgcv within 1e-6.
3. **Example models**: Insurance pricing (Tweedie), mtcars (Gaussian), etc.
4. **CI/CD**: Automated test suite on every commit.

## Development Checklist

Before marking a step complete:
- ✓ Code is production-grade (no stubs, full type hints).
- ✓ Docstrings include mathematical notation and references.
- ✓ Unit tests pass; code coverage >95%.
- ✓ Numerical validation against mgcv within 1e-6.
- ✓ Module is integrated into package `__init__.py`.
- ✓ CHANGELOG updated with step summary.

## Invocation

Use the **pymgcv-gam-architect** agent for:
- Implementing specific steps or phases.
- Architectural decisions and design reviews.
- Validation strategies and testing approaches.
- Mathematical formulation guidance.

Example: `@pymgcv-gam-architect Step 2: Implement thin plate regression splines`

## Next Steps

1. Initialize package structure (setup.py, pyproject.toml, __init__.py).
2. Begin Phase 1: Steps 1–5 (Foundations).
3. Build Phase 2 once Phase 1 is validated.
4. Iterate through all 4 phases.
5. Final validation: comprehensive mgcv comparison on diverse models.
