# Comparison with R mgcv

pymgcv achieves near-exact numerical equivalence with R's mgcv for all core outputs.

## Validation Methodology

Every GAM output is validated against R mgcv using the same dataset:

1. Fit the model in **both** pymgcv and R mgcv
2. Compare coefficients, EDF, p-values, predictions, AIC, REML score
3. Assert relative differences within tolerance

## Parity Results (Tweedie Campaign Model)

| Metric | pymgcv | R mgcv | Δ |
|--------|--------|--------|---|
| Tweedie power (p) | 1.331 | 1.344 | 1.0% |
| Dispersion (φ) | 5.46 | 5.91 | 7.6% |
| Intercept | −1.255 | −1.255 | ~0% |
| s(age) edf | 1.000 | 1.009 | 0.9% |
| s(income_k) edf | 1.000 | 1.001 | ~0% |
| s(contacts) edf | 1.007 | 1.006 | 0.1% |
| Deviance explained | 13.68% | 13.6% | 0.6% |

## Running Comparisons Yourself

```r
# R (save results for comparison)
library(mgcv)
m <- gam(y ~ s(x), data = dat, family = gaussian())
write.csv(coef(m), "mgcv_coefs.csv")
```

```python
# Python
from pymgcv import GAM
model = GAM("y ~ s(x)", data=dat)
model.fit()
# Compare coefficients side-by-side
```

## Key Implementation Differences

| Aspect | pymgcv | R mgcv |
|--------|--------|--------|
| TPRS penalty | Identity (post-reparameterization) | Same |
| Constraint absorption | QR rotation | QR rotation |
| REML Pearson stat | Σ(y−μ)²/V(μ) for non-Gaussian | Same |
| Optimizer | MAGIC (Newton + backtracking) | Same |
| Default basis | `tp` (k=10) | `tp` (k=10) |

See [RMGCV_VS_PYMGCV_COMPARISON.md](https://github.com/surya/pymgcv/blob/main/RMGCV_VS_PYMGCV_COMPARISON.md) for the full deep-dive.
