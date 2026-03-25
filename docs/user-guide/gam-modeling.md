# GAM Modeling Guide

## Smooth Term Types

pymgcv supports all major smooth basis types from R's mgcv:

| Code | Name | Description |
|------|------|-------------|
| `tp` / `tprs` | Thin plate regression spline | Default. Optimal smooth in an MSE sense |
| `cr` | Cubic regression spline | Knot-based, local support |
| `cs` | Cubic shrinkage spline | Cubic with extra shrinkage penalty |
| `bs` | B-spline | Flexible, configurable order |
| `ps` | P-spline | Eilers–Marx penalized B-spline |
| `cc` / `cp` | Cyclic cubic / P-spline | For periodic data |
| `re` | Random effect | Gaussian random intercept/slope |
| `gp` | Gaussian process | GP-based smooth |
| `ad` | Adaptive smooth | Spatially varying penalty |
| `fs` | Factor smooth | Smooth-by-factor interaction |
| `te` / `ti` / `t2` | Tensor products | Multidimensional smooths |

### Usage in Formulas

```python
from pymgcv import GAM

# Thin plate (default)
model = GAM("y ~ s(x)", data=df)

# Cubic spline with 20 knots
model = GAM("y ~ s(x, bs='cr', k=20)", data=df)

# Tensor product of two variables
model = GAM("y ~ te(x1, x2)", data=df)

# By-variable (varying coefficient)
model = GAM("y ~ s(x, by=group)", data=df)

# Fixed df (no smoothing penalty)
model = GAM("y ~ s(x, fx=True, k=5)", data=df)
```

## Distribution Families

| Family | Link | Use case |
|--------|------|----------|
| Gaussian | identity | Continuous outcomes |
| Poisson | log | Count data |
| Binomial | logit | Binary / proportion data |
| Gamma | log | Positive continuous (skewed) |
| Tweedie | log | Insurance loss costs, zero-inflated |
| Inverse Gaussian | 1/μ² | Highly skewed positive |
| Negative Binomial | log | Overdispersed counts |

```python
# Poisson GAM
model = GAM("count ~ s(x)", family="poisson", data=df)

# Tweedie with power estimation
from pymgcv import Tweedie
model = GAM("y ~ s(x)", family=Tweedie(estimate_power=True), data=df)
```

## Smoothing Parameter Estimation

pymgcv supports three criteria (matching mgcv):

- **REML** (default) — Restricted Maximum Likelihood
- **GCV** — Generalized Cross-Validation
- **AIC/UBRE** — Unbiased Risk Estimator

The MAGIC optimizer estimates smoothing parameters using Newton's method
with analytical gradients and Hessians of the REML objective.

## Model Comparison

```python
from pymgcv import anova_gam, compare_models, aic, bic

# ANOVA-style comparison
anova_gam(model1, model2)

# AIC / BIC table
compare_models(model1, model2, model3)
```

## Prediction

```python
# Predict on new data
preds = model.predict(new_df, scale="response")

# Link scale
preds_link = model.predict(new_df, scale="link")
```
