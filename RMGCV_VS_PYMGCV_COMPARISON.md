# R mgcv vs pymgcv — Key Calculation Differences

## Summary

pymgcv is a Python re-implementation of R's `mgcv` package for fitting Generalized Additive Models (GAMs). While both produce statistically equivalent results for practical use, there are implementation-level differences that lead to minor numerical deviations. This document explains **what** differs, **why**, and the **impact**.

---

## Output Comparison (Campaign Tweedie Model)

| Metric | pymgcv | R mgcv | Δ | Material? |
|--------|--------|--------|---|-----------|
| Tweedie power (p) | 1.331 | 1.344 | 1.0% | No |
| Dispersion (φ) | 5.46 | 5.91 | 7.6% | No (driven by p) |
| Intercept | −1.255 | −1.255 | ~0% | No |
| s(age) edf | 1.000 | 1.009 | 0.9% | No |
| s(income_k) edf | 1.000 | 1.001 | ~0% | No |
| s(contacts) edf | 1.007 | 1.006 | 0.1% | No |
| s(age) F-statistic | 306 | 282 | 8.5% | No |
| s(income_k) F-statistic | 269 | 246 | 9.3% | No |
| s(contacts) F-statistic | 91 | 84.6 | 7.5% | No |
| Deviance explained | 13.68% | 13.6% | 0.6% | No |
| Total EDF | 4.01 | 4.02 | ~0% | No |
| Mean prediction | 0.3965 | 0.3972 | 0.02% | No |

---

## Key Calculation Differences

### 1. TPRS Basis Construction

| Aspect | R mgcv | pymgcv | Impact |
|--------|--------|--------|--------|
| **Penalty matrix** | Identity (after eigen-reparameterization: divide range columns by √D_k) | Identity (same reparameterization implemented) | ✅ Matched |
| **Constraint absorption** | QR-based (`absorb.cons`): rotate parameter space via QR of constraint vector | QR-based: identical `Q[:,1:]` rotation of both B and S | ✅ Matched |
| **Kernel function** (d=1,m=2) | `r³` (unnormalized) | `r³` (unnormalized) | ✅ Matched |
| **Eigendecomposition** | Fortran LAPACK `dsyev` | SciPy `linalg.eigh` (calls same LAPACK) | ✅ Matched |
| **Knot selection** (n>2000) | Space-filling via `mgcv:::mini.roots` | Quantile-based (1d) / k-means (multi-d) | Minor — different knot placement may cause tiny basis differences for large datasets |

**Why it matters**: The reparameterization ensures all wiggly basis components are penalized equally by a single λ. Without it, a smooth like `s(income_k)` with eigenvalue spread of 9.1M would have some range columns barely penalized, artificially inflating EDF.

---

### 2. REML Objective (Smoothing Parameter Selection)

| Aspect | R mgcv | pymgcv | Impact |
|--------|--------|--------|--------|
| **Gaussian REML** | Profiled: `n_eff·log(RSS_p/n_eff) + log|A| − log|S⁺|` where `RSS_p = ‖y−Xβ‖² + β'Sβ` | Identical formula | ✅ Matched |
| **Non-Gaussian REML** | Profiled working-model REML: uses **Pearson χ²** = `Σ(y−μ)²/V(μ)` as the "RSS" in the profiled formula | Pearson χ² = `Σ(y−μ)²/V(μ)` (same formula) | ✅ Matched |
| **Dispersion during λ optimization** | φ = 1 throughout (profiled out analytically) | φ = 1 throughout | ✅ Matched |
| **Gradient formula** | `λⱼ·n_eff·β'Sⱼβ/RSS_p + λⱼ·tr(A⁻¹Sⱼ) − rank_j` | Same formula | ✅ Matched |
| **Hessian** | Expected Hessian + negative semi-definite correction from fit term | Same: `λⱼλₖ·tr(A⁻¹Sⱼ·A⁻¹Sₖ) − fit_j·fit_k/(n_eff·RSS_p)` | ✅ Matched |
| **Numerical implementation** | Fortran pivoted Cholesky; sparse matrix support | Python: Cholesky → ridge fallback → SVD (PenalizedSolver) | Minor numerical precision differences |

**Why it matters**: Using Pearson χ² instead of deviance is critical for non-canonical links (e.g., log link with Tweedie). The Pearson statistic weights observations by the variance function, pushing λ higher when data has high dispersion — correctly matching R's smooth-start behavior.

---

### 3. Tweedie Power Estimation

| Aspect | R mgcv | pymgcv | Impact |
|--------|--------|--------|--------|
| **Outer search** | Grid + Brent optimization over p ∈ (1,2) | `scipy.optimize.minimize_scalar` (Brent, bounded) with `xatol=1e-4` | Minor — both use Brent-type methods |
| **Inner objective** | R's exact restricted log-likelihood from the PIRLS working model | Full Laplace-approximate REML: `−2ℓ(β̂,φ̂;p) + β̂'Sβ̂/φ̂ + log|A| − p·log(φ̂) − log|S⁺|` | **This causes the p=1.331 vs 1.344 gap** |
| **Wright function** | R's `ldTweedie` C implementation with series expansion | Python's `scipy.special` + series expansion with caching | Minor — different series truncation points |
| **φ estimation in power search** | From PIRLS working model variance | Pearson residuals: `φ̂ = Σ(y−μ)²/V(μ,1) / (n−edf)` | ✅ Equivalent |
| **Warm-starting** | Each p evaluation warm-starts from previous λ | Each p evaluation warm-starts from the initial model's λ | ✅ Same approach |

**Why it matters**: The 1% gap in estimated power (1.331 vs 1.344) is the **primary driver** of all other numerical differences. The dispersion (φ = 5.46 vs 5.91) and F-statistics (8–9% gap) flow directly from this. The gap arises because R evaluates the exact restricted log-likelihood from within its Fortran PIRLS machinery, while pymgcv evaluates a Laplace approximation to the marginal likelihood with the Wright function computed externally.

---

### 4. Lambda Initialization

| Aspect | R mgcv | pymgcv | Impact |
|--------|--------|--------|--------|
| **Gaussian** | Initial sp from eigen-balancing heuristics | `λ = σ²(y)/n` (noise-per-observation scale) | ✅ Both find the same REML minimum |
| **Non-Gaussian** | Initial sp from first PIRLS iteration | `λ = ‖X'WX‖/‖S‖` at null model (μ = g⁻¹(offset)) | ✅ Both reach the same converged λ |
| **Why different inits work** | R's Fortran optimizer explores broadly | With identity penalty, REML has a bimodal landscape for Gaussian data; initialization must be below the barrier | No impact on final result |

---

### 5. F-Statistics & p-values (Summary Table)

| Aspect | R mgcv | pymgcv | Impact |
|--------|--------|--------|--------|
| **Bayesian posterior covariance** | `Vβ = (X'WX + Sλ)⁻¹` at φ = 1 for IRLS weights | Same: φ = 1 in IRLS weights | ✅ Matched |
| **Wald F-test** | Wood (2013) — `F = β̂'ₛ Vₛ⁻¹ β̂ₛ / edf` with reference df from eigenvalues of Vₛ | Same Wood (2013) formulation | ✅ Matched |
| **Final φ for reporting** | Estimated from Pearson residuals after convergence | Same: `φ̂ = Σ(y−μ)²/V(μ) / (n−Σedf)` | ✅ Matched |
| **F-stat differences** | — | 8–9% higher than R | Driven by p difference (1.331 vs 1.344) changing the variance function and therefore the posterior covariance |

---

### 6. EDF Computation

| Aspect | R mgcv | pymgcv | Impact |
|--------|--------|--------|--------|
| **Formula** | `edf = tr((X'WX + Sλ)⁻¹ X'WX)` at φ = 1 | Same formula | ✅ Matched |
| **Per-smooth EDF** | Extracted from block-diagonal trace | Same: `tr(A⁻¹ · X'ₛWXₛ)` per smooth block | ✅ Matched |
| **Result** | age=1.009, income=1.001, contacts=1.006 | age=1.000, income=1.000, contacts=1.007 | Near-exact — sub-1% differences |

---

## Root Cause Chain

```
p estimation (1.331 vs 1.344)  ← Laplace REML approx vs R's exact restricted loglik
       ↓
φ estimation (5.46 vs 5.91)    ← φ = Pearson/df, and V(μ) = μ^p changes with p
       ↓
F-statistics (8-9% higher)     ← F depends on posterior covariance, which uses V(μ)
       ↓
All other metrics match within 1%
```

The single root cause is the **Tweedie power estimation** using a Laplace approximation to the marginal likelihood vs R's exact restricted log-likelihood. This 1% difference in p cascades into the dispersion and F-statistics. The EDF, intercept, deviance explained, and predictions are near-identical.

---

## When These Differences Matter

| Use Case | Material? | Recommendation |
|----------|-----------|----------------|
| Campaign targeting / segmentation | **No** | Use pymgcv directly |
| Insurance pricing (bulk portfolio) | **No** | Models are statistically equivalent |
| Tail-sensitive pricing (high-value policies) | **Possibly** | Validate on tail quantiles |
| Regulatory / audit (exact R replication required) | **Yes** | Use R mgcv directly |
| Academic publication | **No** | Report implementation and note minor p difference |

---

## Technical Implementation Notes

| Component | File | Key Implementation Detail |
|-----------|------|--------------------------|
| TPRS basis + identity penalty | `pymgcv/smooth/thin_plate.py` | Range columns divided by `√|D_k|`; QR constraint via `Z_con = Q[:,1:]` |
| Profiled REML (Gaussian) | `pymgcv/optimizer/reml_objective.py` | `n_eff·log(RSS_p/n_eff) + logdet_A − γ·log_S_plus` |
| Profiled REML (non-Gaussian) | `pymgcv/optimizer/reml_objective.py` | Same formula with Pearson χ² replacing RSS |
| Power search | `pymgcv/api/gam.py` | Full Laplace REML with Wright function for cross-p comparison |
| λ initialization | `pymgcv/optimizer/magic_optimizer.py` | Gaussian: `σ²/n`; Non-Gaussian: `‖X'WX‖/‖S‖` |
| Summary F-stats | `pymgcv/api/summary.py` | Wood (2013) Wald test with φ=1 IRLS weights |
