# PyMGCV Implementation Roadmap to 98% Parity

**Created:** March 16, 2026  
**Current Parity:** 46.7/100 (47%)  
**Target Parity:** 98/100 (98%)  
**Gap:** 51.3 points  
**Estimated Total Effort:** 92–110 hours  

---

## EXECUTIVE SUMMARY

### Current Scoring Breakdown

```
Families:       64.3/100  (Gap: 35.7)
Smooth Bases:   34.7/100  (Gap: 65.3)  ⚠️ CRITICAL
Optimization:   45.0/100  (Gap: 55.0)  ⚠️ CRITICAL
Inference:      50.0/100  (Gap: 50.0)
Diagnostics:    43.8/100  (Gap: 56.2)  ⚠️ CRITICAL
Specification:  35.0/100  (Gap: 65.0)  ⚠️ CRITICAL

OVERALL:        46.7/100  ΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔΔ 98.0/100
```

### Key Insight: Strategic Order Matters

The roadmap is sequenced to maximize **cumulative progress** through **dependencies**:
- Early wins build foundation for later features
- Parallel workstreams reduce total time
- Testing can begin immediately after each component

---

## PART I: HIGHEST-IMPACT IMPROVEMENTS (15 CORE TASKS)

### 1️⃣ PRIORITY TIER 1: BLOCKING ISSUES (Must First)
These unlock multiple downstream features.

---

### Task 1: Fix PIRLS Solver Numerical Stability (Non-Gaussian Families)

**Current Status:** 
- PIRLS solver in `pymgcv/optimizer/pirls.py` works for Gaussian/Poisson
- Binomial, Negative Binomial, Inverse Gaussian families produce NaN/Inf in weights
- Families are correctly implemented (unit tests pass), but solver fails ~2 iterations in

**What's Missing vs MGCv:**
- R mgcv handles all families robustly via stabilized IRLS
- PyMGCV crashes on ill-conditioned weight calculations
- No backtracking or damping strategy for non-Gaussian

**Implementation Approach:**
```
Step 1: Add Numerical Safeguards (~30 lines)
  - Clamp weights to [1e-10, 1e10] range
  - Detect zero/infinite derivatives (dmu/deta)
  - Use -max(mu-epsilon, epsilon) logit handling
  - Add 1e-12 regularization to diagonal W matrix

Step 2: Implement Damping/Step Halving (~40 lines)
  - Detect loss increase between iterations
  - Apply step-size reduction λ ∈ [0.5, 1.0]
  - Use quadratic interpolation to find optimal step
  - Revert to previous iterate if no improvement

Step 3: Add Convergence Safeguards (~20 lines)
  - Monitor relative change in coefficients
  - Detect divergence patterns
  - Graceful failure with informative message
```

**Estimated Effort:** 4–5 hours (code + testing)

**Expected Score Gain:**
- Families: +15 points (45→60 binomial, 40→55 nb, 40→55 ig)
- **Total gain: +15 points**

**Testing:**
```python
# Test binomial family fitting
y_binom = np.random.binomial(1, 0.3 + 0.4*np.sin(x))
model = GAM('y ~ s(x)', family='binomial').fit()
assert np.allclose(model.coefficients, expected, atol=1e-4)

# Test Negative Binomial
y_nb = np.random.negative_binomial(2, 0.4, 100)
model = GAM('y ~ s(x)', family='negative_binomial').fit()

# Test Inverse Gaussian
y_ig = np.random.gamma(2, scale=0.5, size=100)**(-1)
model = GAM('y ~ s(x)', family='inverse_gaussian').fit()
```

**Dependencies:** None (standalone solver improvement)

**Parallel? Yes** - Can start immediately while other tasks proceed

---

### Task 2: Implement Complete Tensor Product Smooths (te/ti/t2)

**Current Status:**
- Only TPRS (univariate) available
- Tensor products: 0/100 score
- Framework exists in sparse comments but no implementation

**What's Missing vs mgcv:**
- R mgcv has:
  - `te(x, y)` - full tensor product
  - `ti(x, y)` - tensor product of contrasts (removes marginal effects)
  - `t2(x, y)` - alternative tensor interaction
- PyMGCV has: nothing

**Impact:** Tensor products are **essential for 2D/3D smoothing** (~40% of real-world GAM use)

**Implementation Approach:**

```
Step 1: Univariate Margin Extraction (~80 lines)
  - For each dimension, construct marginal basis
    e.g., tprs(x), tprs(y)
  - Extract penalty matrices for each
  - Store basis functions & dimensions

Step 2: Full Tensor Product Assembly (~100 lines)
  - Kronecker product of basis matrices: X = X_x ⊗ X_y
  - Kronecker sum for penalties: P = P_x ⊗ I_y + I_x ⊗ P_y
  - Handle scaling (divide by product of dimensions)
  - Verify design matrix shape: n × (d_x * d_y)

Step 3: Tensor of Contrasts (ti) (~80 lines)
  - Decompose each margin into scaling + deviations
  - Keep only interaction (deviation ⊗ deviation) terms
  - Useful for varying-coefficient models
  - Reduces main effects overlap

Step 4: Alternative Tensor (t2) Parametrization (~60 lines)
  - Use eigendecomposition-based scaling
  - Interaction scaling preserves smoothness direction
  - Simpler penalty diagonal structure
  - Compare AIC/GCV vs te()

Step 5: Integration with Design Matrix (~80 lines)
  - Update ModelMatrix to handle tensor syntax
  - Parse `te(x, y, fx=c(FALSE, TRUE),` etc.
  - Handle optional fixed effects flags
  - Implement basis dimension control
```

**Estimated Effort:** 12–15 hours (design + implementation + testing)

**Expected Score Gain:**
- Smooth Bases: +25–30 points (0→25–30 tensor)
- **Total gain: +25–30 points**

**Testing:**
```python
# Test 2D surface
X = np.random.randn(100, 2)
y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + np.random.randn(100)*0.1

model = GAM('y ~ te(x1, x2)', family='gaussian').fit()
model_r = mgcv_equivalent('y ~ te(x1, x2)')

# Compare coefficients, EDF, GCV
assert np.allclose(model.coef, model_r.coef, atol=1e-4)
assert np.abs(model.edf[0] - model_r.edf[0]) < 0.5

# Test ti variant
model_ti = GAM('y ~ ti(x1, x2)', family='gaussian').fit()
# Should have fewer parameters than te()
assert len(model_ti.coef) < len(model.coef)
```

**Dependencies:** 
- Requires working PIRLS solver (Task 1)
- Robust design matrix assembly

**Parallel? Partially** - Can design while Task 1 is being tested; implementation after Task 1 complete

---

### Task 3: Implement by-Variable Support (Varying-Coefficient Models)

**Current Status:**
- Score: 0/100 (completely missing)
- Formula parser accepts smooth syntax but doesn't recognize `by` parameter
- Not integrated into design matrix assembly

**What's Missing vs mgcv:**
- R mgcv: `s(x, by=group)` produces separate smooth for each group level
- `by` creates factor interaction: smooth * factor
- Essential for:
  - Separate smooths per treatment group
  - Spatial heterogeneity models
  - Multi-site studies with site-specific effects

**Implementation Approach:**

```
Step 1: Formula Parser Extension (~40 lines)
  - Recognize "by=" parameter in smooth spec
  - Extract factor/continuous variable name
  - Store in smooth spec object
  - Validate that by-variable exists in data

Step 2: Design Matrix Expansion (~100 lines)
  - For s(x, by=group) where group has k levels:
    - Create k separate basis matrices (one per level)
    - Expand full design matrix: [B_1 | B_2 | ... | B_k]
    - Padding with zeros: B_i[group≠i] = 0
    - Shape: n × (k * p) where p = basis dimension
  - For by with continuous variable:
    - Element-wise multiply: [x_cont * B_1, x_cont * B_2, ...]

Step 3: Penalty Matrix Expansion (~80 lines)
  - Kronecker structure for by-penalties
  - Each level gets separate penalty same structure
  - Block-diagonal penalty: diag(P, P, ..., P)
  - Smoothing parameter can be shared or separate

Step 4: Model Matrix Integration (~60 lines)
  - Update ModelMatrix.assemble() for by-variables
  - Handle mixed scenarios (smooth + by, linear + by)
  - Proper indexing into coefficient vector
  - Extraction for prediction/summary

Step 5: Prediction & Interpretation (~70 lines)
  - Separate predictions per level
  - Bootstrapped confidence bands per group
  - Summary statistics grouped by factor level
  - Visualization: overlaid smooths per group
```

**Estimated Effort:** 10–12 hours (implementation + testing + validation)

**Expected Score Gain:**
- Specification: +20–25 points (0→20–25 by-variables)
- Inference: +5 points (better summary/prediction)
- **Total gain: +25–30 points**

**Testing:**
```python
# Test factor by-variable
df = pd.DataFrame({
    'x': np.random.randn(200),
    'y': np.random.randn(200),
    'group': np.repeat(['A', 'B', 'C'], 67)[:200]
})
model = GAM('y ~ s(x, by=group)', data=df).fit()

# Compare with R
# expect 3 separate smooths, edf should reflect independence
assert len(model.edf_per_smooth['s(x, by=group)']) == 3

# Test continuous by-variable
df['weight'] = np.random.rand(200)
model2 = GAM('y ~ s(x, by=weight)', data=df).fit()
# expect amplitude of smooth to scale with weight
```

**Dependencies:**
- Requires Task 1 (stable solver) if using non-Gaussian
- Enhanced formula parser (existing code, minimal changes)
- Design matrix assembly (existing, needs extension)

**Parallel? Yes** - Can design/implement in parallel with Tasks 1-2

---

### Task 4: Implement Weights Support

**Current Status:**
- Score: 0/100 (completely missing)
- PIRLS solver doesn't use case weights
- ModelMatrix doesn't extract/store weights

**What's Missing vs mgcv:**
- R mgcv: `gam(..., weights=w)` scales PIRLS objective
- Weighted log-likelihood: $\sum w_i \ell_i$
- Weighted penalty: $\boldsymbol{\beta}^T \mathbf{P} \boldsymbol{\beta} / \sum w_i$
- Essential for:
  - Robust regression (downweight outliers)
  - Survey data (design weights)
  - Heteroscedastic fitting

**Implementation Approach:**

```
Step 1: Data Extraction (~20 lines)
  - Check for 'weights=' parameter in GAM.__init__()
  - Load weights vector from data
  - Validate: weights > 0, finite
  - Normalize to sum = n (optional, for interpretability)

Step 2: PIRLS Integration (~50 lines)
  - Modify weight calculation: w *= weights
  - Update working variable: z *= sqrt(weights)
  - Update design matrix: X *= sqrt(weights) (row-wise)
  - Update response: y *= sqrt(weights)
  - Equivalent: solving (sqrt(W) X)β = sqrt(W) z

Step 3: Penalty Modification (~30 lines)
  - Weighted penalty: λ(P / mean(weights))
  - Ensures same smoothing relative to weight magnitude
  - Alternative: λ(P * mean(weights)) for fixed-effects view

Step 4: EDF Adjustment (~30 lines)
  - Hat matrix: H = X(X'WX)^{-1}X'W
  - EDF = trace(H)
  - Confidence intervals: adjust for weight variation

Step 5: Summary & Diagnostics (~40 lines)
  - Residuals: weighted standardization
  - Leverage: weighted influence
  - Plots: bubble size by weight
```

**Estimated Effort:** 6–8 hours (implementation + validation)

**Expected Score Gain:**
- Specification: +15–20 points (0→15–20 weights)
- Diagnostics: +2–3 points (better residuals)
- **Total gain: +17–23 points**

**Testing:**
```python
# Test with case weights
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'weights': np.random.exponential(1, 100)
})

model_unweighted = GAM('y ~ s(x)', data=df).fit()
model_weighted = GAM('y ~ s(x)', data=df, weights='weights').fit()

# Coefficients differ
assert not np.allclose(model_unweighted.coef, model_weighted.coef)

# Compare with R for numerical accuracy
```

**Dependencies:**
- Task 1 (stable PIRLS) recommended but not required (Gaussian works)
- Design matrix assembly (existing)

**Parallel? Yes** - Independent of Tasks 2-3

---

### Task 5: Complete Cubic Regression Spline Implementation

**Current Status:**
- Score: 35/100 (partial implementation)
- `pymgcv/smooth/cubic_spline.py` has skeleton framework
- Basis matrix generation incomplete
- Not validated against mgcv

**What's Missing vs mgcv:**
- R mgcv cubic regression spline:
  - Knots auto-placed (quantiles of predictor)
  - Basis: cubic polynomial at each knot interval
  - Penalty: integral of squared second derivative
  - Effective degrees of freedom calculation
- PyMGCV: Framework only, no working implementation

**Implementation Approach:**

```
Step 1: Knot Placement (~40 lines)
  - Auto knots: quantiles(x, number.of.knots)
  - Typical default: max(3, floor(sqrt(n)))
  - Option for fixed knots
  - Boundary knots at min/max of x

Step 2: Cubic Basis Construction (~100 lines)
  - Create cubic polynomial basis at each knot interval
  - Use spline.CubicSpline from scipy.interpolate
  - OR implement manually: cubic Hermite polynomials
  - Basis dimension = num_knots + 2
  - Normalize basis (optional, for stability)

Step 3: Penalty Matrix (~80 lines)
  - Construct integrated squared second derivative
  - Semi-definite matrix: P = ∫ S''(x)^T S''(x) dx
  - Numerical integration: Gauss quadrature over knot intervals
  - Check rank deficiency (should be 3 for natural cubic)
  - Eigen-truncation if needed

Step 4: Knot Adaptation Strategy (~60 lines)
  - User-specified knot numbers
  - Data-adaptive knot placement
  - Grouped knots for clusters
  - Knot redundancy detection

Step 5: Validation & Testing (~80 lines)
  - Compare basis shape with R mgcv output
  - Verify penalty matrix properties
  - Check TPRS equivalence (should match with many knots)
  - EDF convergence as knots increase
```

**Estimated Effort:** 10–12 hours (implementation + validation)

**Expected Score Gain:**
- Smooth Bases: +15–20 points (35→50–55 cubic)
- **Total gain: +15–20 points**

**Testing:**
```python
# Test cubic spline vs TPRS
x = np.linspace(0, 1, 100)
y = np.sin(x) + np.random.randn(100)*0.1

# Many knots cubic should approach TPRS
model_cubic = GAM('y ~ s(x, bs="cr", k=30)').fit()  # 30 knots
model_tprs = GAM('y ~ s(x, bs="tp", k=30)').fit()   # TPRS

# Should have similar fit quality
assert np.abs(model_cubic.dev - model_tprs.dev) / model_tprs.dev < 0.05

# Validate against R output
```

**Dependencies:**
- Task 1 (solver) for non-Gaussian (optional, cubic works fine with Gaussian)
- Design matrix assembly (existing)

**Parallel? Yes** - Independent of Tasks 2-4

---

### Task 6: Implement Fixed Smoothing Parameters (sp= specification)

**Current Status:**
- Score: Included in "formula parsing" (65/100) but not implemented
- GAM class doesn't recognize/handle `sp=` parameter
- Always runs MAGIC optimization

**What's Missing vs mgcv:**
- R mgcv: `gam(..., sp=c(0.1, 1.0, ...))` fixes smoothing params
- User can disable MAGIC optimizer entirely
- Essential for:
  - Reproducible results (fixed parameters)
  - Grid searches over λ
  - Theory validation (known λ values)
  - Comparison with other methods

**Implementation Approach:**

```
Step 1: GAM Class Parameter Addition (~20 lines)
  - Add sp= parameter to __init__()
  - If sp provided, skip MAGIC optimization
  - Validate: len(sp) == num_smooth_terms
  - Store as self.sp_fixed

Step 2: MAGIC Bypass (~30 lines)
  - Check if sp is provided
  - If yes:
    - Use provided values directly
    - Skip entire optimization loop
    - Set initial smoothing_parameters = sp
  - If no: proceed with MAGIC (existing code)

Step 3: Coefficient Fitting (~20 lines)
  - Run single PIRLS iteration with fixed λ
  - Solve: (X'WX + Σ λ_j P_j) β = X'Wz
  - Return coefficients, no optimization

Step 4: EDF & Statistics Calculation (~30 lines)
  - Compute EDF with fixed λ
  - Hat matrix: H = X(X'WX + Σ λ_j P_j)^{-1}X'W
  - Confidence intervals unchanged
  - P-values now from fixed model

Step 5: Documentation & Examples (~40 lines)
  - Clear docs: when to use fixed sp
  - Example: grid search over sp
  - Example: reproducibility use case
```

**Estimated Effort:** 3–4 hours (simple implementation)

**Expected Score Gain:**
- Specification: +5–8 points (toward formula completeness)
- Optimization: +3–5 points (alternative method)
- **Total gain: +8–13 points**

**Testing:**
```python
# Test fixed smoothing parameters
model_auto = GAM('y ~ s(x1) + s(x2)').fit()
sp_auto = model_auto.smoothing_parameters

# Fix to same values
model_fixed = GAM('y ~ s(x1) + s(x2)', sp=sp_auto).fit()

# Should produce identical coefficients (approximately)
assert np.allclose(model_fixed.coef, model_auto.coef, atol=1e-6)

# Test grid search
sp_grid = [(0.01, 0.01), (0.1, 0.1), (1.0, 1.0)]
results = []
for sp in sp_grid:
    m = GAM('y ~ s(x1) + s(x2)', sp=sp).fit()
    results.append(m.gcv)
    
# Find best
```

**Dependencies:** None (minimal integration)

**Parallel? Yes** - Can implement immediately

---

## 2️⃣ PRIORITY TIER 2: HIGH-IMPACT INFRASTRUCTURE (Parallel with Tier 1)

These can start while Tier 1 tasks are being tested.

---

### Task 7: Complete B-Spline Implementation

**Current Status:**
- Score: 40/100 (framework exists)
- `pymgcv/smooth/bspline.py` has multiple partial implementations
- Not fully integrated or tested

**What's Missing vs mgcv:**
- R mgcv B-spline basis:
  - De Boor algorithm reference implementation
  - Periodic/closed-form bases
  - Multiple penalty orders (0, 1, 2 for smoothness)
  - Knot sequence control
- PyMGCV: Multiple stubs, unclear which is current

**Implementation Approach:**

```
Step 1: De Boor Algorithm Implementation (~80 lines)
  - Standard B-spline basis evaluation
  - Recursive formula for arbitrary degree
  - Efficient cumulative evaluation
  - Handle knot multiplicities (repeated knots)

Step 2: Knot Sequence Management (~60 lines)
  - Default: equally-spaced knots
  - Uniform knot vector construction
  - Clamped knots (natural boundary conditions)
  - Periodic knots for cyclic data

Step 3: Basis Matrix Assembly (~100 lines)
  - Evaluate All B-splines at design data
  - Shape: n × (num_knots + degree)
  - Handle boundary issues
  - Orthonormalization option

Step 4: Penalty Matrices (0, 1, 2 order) (~120 lines)
  - Order 0: difference penalty (Eilers-Marx)
  - Order 1: integrated first derivative
  - Order 2: integrated second derivative (roughness)
  - Construction via finite differences or quadrature

Step 5: Comparison & Validation (~80 lines)
  - Verify basis against mgcv output
  - Compare penalties (within numerical tolerance)
  - Test convergence as basis dimension varies
```

**Estimated Effort:** 12–15 hours (implementation + validation)

**Expected Score Gain:**
- Smooth Bases: +15–20 points (40→55–60 bspline)
- **Total gain: +15–20 points**

**Testing:**
```python
# Test B-spline basis
x = np.linspace(0, 1, 100)
bs = BSplineBasis(x, knots=10, degree=3)
X = bs.basis_matrix(x)

# Shape should be (100, 13)  # 10 + degree
assert X.shape == (100, 13)

# Partition of unity: each row sums ~1
assert np.allclose(X.sum(axis=1), 1, atol=1e-10)

# Compare fit with mgcv
```

**Dependencies:**
- Task 1 (solver) for robust fitting
- Smooth base infrastructure (existing)

**Parallel? Yes** - Can design/implement in parallel with Tier 1

---

### Task 8: Implement P-Splines (Penalized B-Splines)

**Current Status:**
- Score: 45/100 (partial)
- Framework exists in `pymgcv/smooth/pspline.py`
- Depends on B-splines (Task 7)

**What's Missing vs mgcv:**
- R mgcv P-spline: Eilers-Marx method
  - B-spline basis + difference penalty combo
  - Avoids knot number selection (uses roughness λ)
  - Automatic "wiggliness" control
  - Very flexible in practice

**Implementation Approach:**

```
Step 1: Basis + Penalty Combo (~60 lines)
  - Use Task 7 B-spline basis
  - Apply difference penalty (default 2nd order)
  - Combined objective: min ||y - Xβ||^2 + λ||Dβ||^2
  - D = difference matrix

Step 2: B-Spline Order Selection (~40 lines)
  - Default: cubic (degree 3)
  - Higher degree → smoother basis
  - Lower degree → more local

Step 3: Knot Number Control (~50 lines)
  - Fewer knots → less flexibility
  - More knots → less bias, more variance
  - Interact with λ: more knots requires higher λ
  - Auto: based on EDF target

Step 4: Smoothing Parameter Optimization (~40 lines)
  - MAGIC optimizer handles P-spline identical to others
  - No special treatment needed (already unified)

Step 5: Diagnostics & Coverage (~50 lines)
  - P-splines should never overfit (penalty prevents it)
  - Coverage of confidence intervals
  - Effective knot count
```

**Estimated Effort:** 6–8 hours (relatively simple given B-spline base)

**Expected Score Gain:**
- Smooth Bases: +10–15 points (45→55–60 pspline)
- **Total gain: +10–15 points**

**Testing:**
```python
# Test P-spline fit
model_ps = GAM('y ~ s(x, bs="ps")').fit()
model_cr = GAM('y ~ s(x, bs="cr")').fit()

# P-spline should be smoother (lower EDF, lower roughness)
assert model_ps.edf_per_smooth['s(x)'][0] <= model_cr.edf_per_smooth['s(x)'][0]
```

**Dependencies:**
- Task 7 (B-spline basis)

**Parallel? Yes** - After Task 7 starts

---

### Task 9: Implement gam.check() Diagnostics

**Current Status:**
- Score: 0/100 (completely missing)
- Residual diagnostics exist (Task is 60/100 via other tools)
- No comprehensive model check tool

**What's Missing vs mgcv:**
- R mgcv `gam.check()` produces:
  - 4-panel diagnostic plot (residuals, qqplot, hist, scale-location)
  - Basis dimension recommendation (increase k if edf ≈ k)
  - Concurvity scores
  - P-value for smooth significance
  - Numerical summary of potential issues

**Implementation Approach:**

```
Step 1: Diagnostic Figure Assembly (~100 lines)
  - 4-panel matplotlib layout:
    Panel 1: Residuals vs fitted (y-hat)
    Panel 2: Q-Q plot (standardized residuals)
    Panel 3: Histogram of residuals
    Panel 4: Scale-location plot (sqrt(|res|) vs fitted)
  - Confidence bands on Q-Q
  - Marginal distributions

Step 2: Basis Dimension Check (~50 lines)
  - For each smooth: compare edf_j to basis dimension k_j
  - Flag if edf_j > 0.9 * k_j (potential oversmoothing)
  - Recommend: increase k by factor 1.5-2
  - Print warning messages

Step 3: Concurvity Analysis (~60 lines)
  - Reuse existing concurvity() calculation
  - Flag high concurvity (>0.8) between terms
  - Recommend re-ordering smooth terms or using ti()
  - Print to console

Step 4: Significance Summary (~40 lines)
  - Print table: smooth term, edf, significance (p-value)
  - Flag marginal (0.05-0.1) and significant (p<0.05) terms
  - Recommend removal of non-significant terms

Step 5: Interactive/Output Options (~50 lines)
  - Return object with diagnostics
  - save_path parameter for saving figure
  - output_level: 'minimal', 'full'
  - Threshold customization
```

**Estimated Effort:** 8–10 hours (implementation + refinement)

**Expected Score Gain:**
- Diagnostics: +25 points (0→25 gam.check)
- Inference: +3–5 points (integrate significance)
- **Total gain: +28–30 points**

**Testing:**
```python
# Test gam.check output
model = GAM('y ~ s(x1) + s(x2) + x3').fit()
check_result = model.gam_check()

# Should have diagnostics for all terms
assert 's(x1)' in check_result['summary']
assert 's(x2)' in check_result['summary']
assert 'x3' not in check_result['summary']  # parametric only

# Visual test
check_result['figure'].savefig('test_gam_check.png')
```

**Dependencies:**
- Residual diagnostics (existing)
- Significance tests (existing)
- Plotting infrastructure (existing)

**Parallel? Yes** - Independent; can implement anytime

---

### Task 10: Improve Confidence Intervals (Better Coverage)

**Current Status:**
- Score: 30/100 (basic delta method exists)
- Intervals use simple delta method: μ ± 1.96*σ
- No consideration of link function curvature
- Limited to gaussian family in practice

**What's Missing vs mgcv:**
- R mgcv:
  - Transforms intervals to response scale (accounts for non-linearity)
  - Bayesian posterior simulation for GLM link transformation
  - Tolerance band option (simultaneous vs pointwise)
  - Heteroscedastic uncertainty (doesn't assume constant SE)

**Implementation Approach:**

```
Step 1: Link-Scale Transformation (~60 lines)
  - Compute intervals in linear predictor space
  - Transform bounds using inverse link function
  - For logit: μ = 1/(1+exp(-η)) → bounds are asymmetric
  - For log: μ = exp(η) → bounds are multiplicative
  - Order-preserve transformed bounds

Step 2: Bayesian Posterior Simulation (~80 lines)
  - Sample β ~ N(β̂, cov(β̂)) [M times, e.g., M=1000]
  - Evaluate linear predictor: η = X[test]*β_sample
  - Transform: μ = g^{-1}(η)
  - Extract percentiles: 0.025, 0.975 per prediction point
  - Better for GLM with nonlinear links

Step 3: Tolerance Bands (Simultaneous CI) (~70 lines)
  - Simultaneous vs pointwise coverage
  - Gateaux critical value correction
  - Bonferroni adjustment
  - Use quantile of Studentized max deviation

Step 4: Heteroscedasticity Handling (~50 lines)
  - Adaptive variance smoothing
  - Local variance estimation near prediction point
  - Residual-based variance model
  - Avoid assumption of constant residual spread

Step 5: Visualization Enhancements (~60 lines)
  - Shaded confidence ribbon (not error bars)
  - Distinction between simultaneous/pointwise
  - Link scale plot option
  - Multiple band widths (68%, 95%, 99.7%)
```

**Estimated Effort:** 10–12 hours (implementation + validation)

**Expected Score Gain:**
- Inference: +15–20 points (30→45–50 confidence_intervals)
- Predictions: +3–5 points (better predict output)
- **Total gain: +18–25 points**

**Testing:**
```python
# Test transformed confidence intervals
df = pd.DataFrame({
    'x': np.linspace(0, 1, 100),
    'y': np.random.binomial(1, 0.3 + 0.4*np.linspace(0, 1, 100))
})
model = GAM('y ~ s(x)', family='binomial', data=df).fit()

pred_df = pd.DataFrame({'x': np.linspace(0, 1, 50)})
pred = model.predict(pred_df, confidence_interval=0.95, se_fit=True)

# Intervals should be in (0, 1) for binomial
assert pred['lower'].min() >= 0
assert pred['upper'].max() <= 1

# Compare with R for accuracy
```

**Dependencies:**
- Task 1 (stable solver) for non-Gaussian
- Prediction infrastructure (existing)

**Parallel? Yes** - Independent

---

## 3️⃣ PRIORITY TIER 3: OPTIMIZATION & SELECTION (Middle Priority)

These improve model fitting quality and alternative algorithms.

---

### Task 11: Complete GCV Criterion Implementation

**Current Status:**
- Score: 50/100 (framework stub exists)
- `pymgcv/criterions/gcv.py` has skeleton
- Not fully functional or tested

**What's Missing vs mgcv:**
- R mgcv GCV:
  - Generalized cross-validation criterion
  - Default/most commonly used method
  - Formula: GCV = n*RSS / (n - EDF)^2
  - Efficient computation via hat matrix trace
- PyMGCV: Framework only

**Implementation Approach:**

```
Step 1: GCV Formula (~30 lines)
  - RSS = sum((y - η̂)^2 * w)  # Weighted residual sum of squares
  - EDF = trace(H)  # Effective degrees of freedom
  - GCV = RSS * n / (1 - EDF/n)^2
  - OR: GCV = n*RSS / (n - EDF)^2
  - Vectorized implementation

Step 2: Gradient Computation (~60 lines)
  - d(GCV)/d(log(λ)) via finite differences
  - Or use automatic differentiation (JAX) if available
  - Validate against numerical gradient

Step 3: Integration with Optimizer (~40 lines)
  - Minimize GCV instead of/in parallel with MAGIC
  - Newton's method: same as MAGIC but different objective
  - Update smoothing parameters iteratively

Step 4: Comparison with MAGIC (~50 lines)
  - Both should produce similar results
  - GCV often favors slightly higher smoothing
  - Compare AIC/REML values post-hoc

Step 5: Validation (~40 lines)
  - Test against R mgcv output
  - Verify criteria values match (within 1%)
  - Check convergence speed
```

**Estimated Effort:** 6–8 hours

**Expected Score Gain:**
- Optimization: +15 points (50→65 gcv)
- **Total gain: +15 points**

**Testing:**
```python
# Test GCV criterion
model = GAM('y ~ s(x1) + s(x2)').fit(criterion='GCV')
gcv_value = model.criterion_value

# Should be minimized
assert gcv_value < (model residual sum of squares)/(model.n)

# Compare with Magic result
model_magic = GAM('y ~ s(x1) + s(x2)').fit(criterion='magic')
# GCV and MAGIC may differ but should be close
```

**Dependencies:**
- Task 1 (solver)
- EDF computation (existing)

**Parallel? Yes** - Can implement independently

---

### Task 12: Complete AIC/UBRE Criterion

**Current Status:**
- Score: 35/100 (partial)
- Framework exists but not fully implemented
- Integration with optimizer incomplete

**What's Missing vs mgcv:**
- R mgcv:
  - AIC: 2*EDF + RSS (information criterion)
  - UBRE: Unbiased Risk Estimator (same as GCV for Gaussian)
  - Used for model comparison and selection
  - Extension to GLM (deviance-based)

**Implementation Approach:**

```
Step 1: AIC Implementation (~40 lines)
  - Gaussian: AIC = 2*k + n*log(RSS/n) where k = EDF
  - Poisson/Binomial: AIC = 2*k + deviance
  - Tweedie: AIC with dispersion adjustment

Step 2: UBRE Implementation (~50 lines)
  - Formula: UBRE = (1 - 2*EDF/n)*RSS + (EDF/n)*σ²
  - Assumes known dispersion σ²
  - Typically similar to GCV

Step 3: Optimization Integration (~40 lines)
  - Newton's method minimizes AIC loss
  - Gradient dAIC/d(log(λ))
  - Alternative to MAGIC (same framework)

Step 4: Model Comparison (~50 lines)
  - Compare AIC across different models
  - Delta AIC: evidence ratio
  - Model averaging weights
```

**Estimated Effort:** 5–7 hours

**Expected Score Gain:**
- Optimization: +15 points (35→50 AIC)
- **Total gain: +15 points**

**Dependencies:**
- Task 1 (solver)
- EDF (existing)

**Parallel? Yes** - With Task 11

---

### Task 13: Implement REML-Based Smoothing Parameter Selection

**Current Status:**
- Score: 40/100 (partial implementation with known bugs)
- `pymgcv/optimizer/reml_objective.py` exists
- Convergence issues reported

**What's Missing vs mgcv:**
- R mgcv REML:
  - Restricted maximum likelihood for λ selection
  - More stable than GCV for small samples
  - Recommended for Poisson/binomial (better λ)
  - Prior-like behavior (Bayesian interpretation)

**Implementation Approach:**

```
Step 1: REML Objective Bug Fixes (~40 lines)
  - Check gradient computation
  - Numerical stability in log-likelihood
  - Handle singular penalties

Step 2: Newton-Raphson Optimization (~60 lines)
  - Hessian computation (second derivative w.r.t. log(λ))
  - Line search for step size control
  - Convergence criteria (gradient norm, parameter change)

Step 3: Bayesian Interpretation (~40 lines)
  - Document Lambda as prior precision
  - EDF as effective sample size
  - Connection to James-Stein shrinkage

Step 4: Comparative Validation (~50 lines)
  - Compare REML vs GCV vs MAGIC
  - Test on standard datasets
  - Document when each is preferred
```

**Estimated Effort:** 8–10 hours (debugging + improvements)

**Expected Score Gain:**
- Optimization: +10–15 points (40→50–55 REML)
- **Total gain: +10–15 points**

**Dependencies:**
- Task 1 (solver)
- EDF (existing)

**Parallel? Yes** - With Tasks 11-12

---

### Task 14: Implement Model Comparison / ANOVA for GAMs

**Current Status:**
- Score: 0/100 (missing)
- No comparison framework

**What's Missing vs mgcv:**
- R mgcv `anova.gam()`:
  - Compare two nested models
  - Test statistic (deviance difference)
  - P-values (χ² or F distribution)
  - Different smoothing parameter scenarios

**Implementation Approach:**

```
Step 1: Comparison Framework (~60 lines)
  - Accept list of fitted GAM models
  - Verify nesting (same data, formula subset)
  - Extract: deviance, EDF, dispersion

Step 2: Test Computation (~80 lines)
  - Deviance difference: D = D_1 - D_2
  - EDF difference: Δ(EDF) = EDF_1 - EDF_2
  - Test statistic: T = D / (dispersion * Δ(EDF))
  - P-value: 1 - F_cdf(T; Δ(EDF), n - EDF_2)

Step 3: Table Output (~60 lines)
  - DataFrame with models, deviance, EDF, test stats
  - Significance stars
  - Pretty printing (mgcv-compatible format)

Step 4: Visualization (~40 lines)
  - Plot deviance traces
  - EDF progression
  - Compare fitted smooths across models
```

**Estimated Effort:** 6–8 hours

**Expected Score Gain:**
- Inference: +10 points (better model assessment)
- **Total gain: +10 points**

**Dependencies:**
- Task 1 (solver)
- Existing model infrastructure

**Parallel? Yes** - Independent

---

## 4️⃣ PRIORITY TIER 4: COMPLETION & POLISH (Lower Priority but Impactful)

---

### Task 15: Implement Cyclic Smooth Bases (cc/cp)

**Current Status:**
- Score: 0/100 (missing)
- No cyclic basis implementation

**What's Missing vs mgcv:**
- R mgcv cyclic smooths:
  - `bs="cc"`: cyclic cubic regression spline
  - `bs="cp"`: cyclic P-spline
  - Auto-wrap boundary (x[max] → x[min])
  - Essential for circular data (months, angles, wind directions)

**Implementation Approach:**

```
Step 1: Cyclic Boundary Conditions (~50 lines)
  - Modify basis so B(x_min) = B(x_max)
  - Penalize discontinuities at boundary
  - Knot placement wraps around

Step 2: Cyclic Cubic Spline (~80 lines)
  - Extend cubic spline (Task 5) with cyclic boundary
  - Knot vector: extends beyond [min, max]
  - Penalty modified for periodicity

Step 3: Cyclic P-Spline (~80 lines)
  - Similar to Task 8 but with cyclic penalty
  - Difference penalty wraps: P_cyc

Step 4: Validation (~50 lines)
  - Test with circular data (wind direction ~[0, 2π])
  - Check smooth continuity at boundary
  - Compare vs. wrapping x to [0, 1] manually
```

**Estimated Effort:** 8–10 hours

**Expected Score Gain:**
- Smooth Bases: +12–15 points (0→12–15 cyclic)
- **Total gain: +12–15 points**

**Dependencies:**
- Task 5 (cubic spline)
- Task 8 (P-spline)

**Parallel? Yes** - After Tasks 5, 8 start

---

### Task 16: Implement Automatic Smooth Selection (LASSO-like smoothing)

**Current Status:**
- Score: Part of "model selection" (not directly scored)
- Framework exists in sparse code
- Not integrated into GAM fitting

**What's Missing vs mgcv:**
- R mgcv `select=TRUE`:
  - Automatic removal of non-significant smooths
  - Shrinkage-like penalty toward zero for unimportant terms
  - AIC/GCV selection among models with/without terms
  - Feature selection via smoothness

**Implementation Approach:**

```
Step 1: Shrinkage Penalty Integration (~60 lines)
  - Add null-space penalty (shrinks toward parametric)
  - Encourage smooth → 0 if data-unsupported
  - Can combine with marginal penalties

Step 2: Model Path Algorithm (~80 lines)
  - Vary shrinkage intensity (lambda for selection)
  - Track which terms remain "active" (edf > 0.01)
  - Find optimal shrinkage via GCV/AIC

Step 3: FDR Control (~50 lines)
  - Multiple testing correction
  - Control false discovery rate across smooths
  - Threshold for keeping term

Step 4: Integration (~40 lines)
  - Add select=True to GAM.__init__()
  - Post-hoc term elimination
  - Report: included/excluded terms
```

**Estimated Effort:** 8–10 hours

**Expected Score Gain:**
- Optimization: +8–10 points (indirect, model selection)
- Inference: +5 points
- **Total gain: +13–15 points**

**Dependencies:**
- Task 1 (solver)
- Task 11 (GCV)

**Parallel? Yes** - Lower urgency

---

## ADDITIONAL ENHANCEMENTS (Not Top 15-16, but Valuable)

### Task 17: Implement Multivariate Predictions & Uncertainty Bands

- **Score impact:** +5–10 points (inference)
- **Effort:** 6–8 hours
- **Approach:** Joint uncertainty bands, simultaneous coverage
- **Testing:** Comparison with mgcv intervals

### Task 18: Enhance Plotting & Visualization

- **Score impact:** +5–8 points (inference visualization)
- **Effort:** 6–8 hours
- **Approach:** mgcv-compatible plot layouts, contour plots, etc.
- **Testing:** Visual comparison with R output

### Task 19: Implement Custom Basis Types (soap, Duchon)

- **Score impact:** +10–15 points (smooth bases)
- **Effort:** 12–15 hours
- **Approach:** Soap film boundaries, Duchon splines
- **Testing:** Specialized domain applications

### Task 20: Add Mixed Model / Random Effects Support

- **Score impact:** +15–20 points (advanced features)
- **Effort:** 20–25 hours
- **Approach:** Smooth term as random effect, variance component estimation
- **Testing:** Comparison with lme4/nlme

---

## PART II: IMPLEMENTATION ROADMAP & SEQUENCING

### Critical Dependencies Map

```
Task 1: PIRLS Stability (4–5 hrs)
  ├─→ Task 5: Cubic Spline (10–12 hrs) ✓
  ├─→ Task 7: B-Spline (12–15 hrs) ✓
  │     └─→ Task 8: P-Spline (6–8 hrs) ✓
  │           └─→ Task 15: Cyclic (8–10 hrs)
  ├─→ Tasks 2,3,4,6: Parallel (30–40 hrs) ✓
  ├─→ Tasks 9,10: Parallel (18–22 hrs) ✓
  └─→ Tasks 11,12,13: Parallel (19–25 hrs) ✓

Task 2: Tensor Products (12–15 hrs) → depends on Task 1 ✓

Legend: ✓ = can run in parallel with other tier 1 tasks
```

### PHASE 1: Foundation & Solver Stability (Week 1)

**Goals:**
- Fix PIRLS for non-Gaussian families
- Enable Tier 2 parallel work

**Tasks:**
1. Task 1: PIRLS Stability (4–5 hrs)

**Outcome:**
- All 7 families can now be robustly fit
- Families score: 64.3 → 79.3 (+15)
- **Overall: 46.7 → 61.7 (+15)**

**Testing:** Verify binomial, NB, IG short runs before proceeding

---

### PHASE 2A: Critical Specification Features (Parallel, Week 1-2)

**Goals:**
- Unlock by-variables and weights (essential for practical models)
- Establish high-impact features early

**Tasks (In Parallel):**
1. Task 3: By-Variables (10–12 hrs)
2. Task 4: Weights (6–8 hrs)
3. Task 6: Fixed sp= (3–4 hrs)

**Effort:** 19–24 hours total (all parallel, ~5 days)

**Outcome:**
- Specification: 35.0 → 60–65 (+25–30)
- **Overall: 61.7 → 86–91 (+25–30)**

---

### PHASE 2B: Basis Completion (Parallel, Week 1-2)

**Goals:**
- Complete missing smooth bases (cubic, bspline, pspline, tensor)
- Massive impact on smooth_bases score

**Tasks (Sequential but Fast):**
1. Task 5: Cubic Spline (10–12 hrs)
2. Task 7: B-Spline (12–15 hrs)
3. Task 8: P-Spline (6–8 hrs)
4. Task 2: Tensor Products (12–15 hrs)
5. Task 15: Cyclic (8–10 hrs) [if time permits]

**Effort:** 48–60 hours total (sequential, ~13 days compressed to ~7 if parallel design)

**Outcome:**
- Smooth Bases: 34.7 → 75–80 (+40–45)
- **Overall: 86–91 → 126–131 (capped at 98)**

---

### PHASE 3: Model Evaluation & Diagnostics (Parallel, Week 2-3)

**Goals:**
- Improve diagnostics and inference
- Enable model criticism and comparison

**Tasks (Parallel):**
1. Task 9: gam.check() (8–10 hrs)
2. Task 10: Confidence Intervals (10–12 hrs)
3. Task 14: Model Comparison (6–8 hrs)

**Effort:** 24–30 hours (parallel, ~3–4 days wallclock)

**Outcome:**
- Diagnostics: 43.8 → 68–73 (+24–30)
- Inference: 50.0 → 63–68 (+13–18)
- **Overall: 126–131 → capped 98 (all sub-components 50+)**

---

### PHASE 4: Optimization Algorithms (Parallel, Week 2-3)

**Goals:**
- Complete smoothing parameter selection methods
- Provide alternative algorithms

**Tasks (Parallel):**
1. Task 11: GCV Completion (6–8 hrs)
2. Task 12: AIC/UBRE (5–7 hrs)
3. Task 13: REML Fixing (8–10 hrs)

**Effort:** 19–25 hours (parallel, ~3–4 days)

**Outcome:**
- Optimization: 45.0 → 65–75 (+20–30)
- Inference: 63–68 → 70–80 (secondary benefit)
- **Overall: Stays near 98 (all sub-components strong)**

---

### PHASE 5: Optional Enhancements (Week 3+)

**If Time Permits:**
- Task 16: Automatic smoothing selection (8–10 hrs)
- Task 17-18: Visualization (6–8 hrs)
- Task 19: Custom bases (12–15 hrs)
- Task 20: Mixed models (20–25 hrs)

---

## PART III: EFFORT ESTIMATES & PROGRESSION

### Summary Effort Table

| Phase | Tasks | Hours | Wallclock | Parallel Potential |
|-------|-------|-------|-----------|-------------------|
| **1: Solver** | Task 1 | 4–5 | 1 day | ~100% |
| **2A: Specification** | Tasks 3,4,6 | 19–24 | 5 days | ~100% |
| **2B: Bases** | Tasks 2,5,7,8,15 | 48–60 | 7–13 days | ~60% (Task 2 →Task 7,8) |
| **3: Inference** | Tasks 9,10,14 | 24–30 | 3–4 days | ~90% |
| **4: Optimization** | Tasks 11,12,13 | 19–25 | 3–4 days | ~90% |
| **5: Optional** | Tasks 16-20 | 50–73 | 8–10 days | 70% |
| **TOTAL (Core 14)** | 1-14 | **114–144** | **14–21 days** | **~70% effective** |

### Parity Progression Targets

```
START:            46.7/100 (47%)
├─ After Phase 1: 61.7/100 (62%)  [+15.0] — 6 days
├─ After Phase 2: 90–95/100 (90–95%) [+28–33] — 18 days
├─ After Phase 3: 96–98/100 (96–98%) [+6–8] — 21 days
├─ After Phase 4: 98+/100 (98%+) [+0–2, capped] — 25 days
└─ After Phase 5: 98+/100 with polish — 35+ days

RECOMMENDED STOP: Phase 3 (21 days, 96–98% parity)
                  Phase 4 polishes to 98% (28 days)
```

---

## PART IV: QUICK WINS vs MAJOR PROJECTS

### Quick Wins (Low Effort, Immediate Payoff)

| Task | Hours | Gain | Priority | ROI |
|------|-------|------|----------|-----|
| Task 6: Fixed sp= | 3–4 | +8–13 | High | 2.5x |
| Task 1: PIRLS Stability | 4–5 | +15 | Critical | 3x |
| Task 4: Weights | 6–8 | +17–23 | High | 2.5x |
| Task 3: By-Variables | 10–12 | +25–30 | Critical | 2.3x |
| Task 9: gam.check() | 8–10 | +28–30 | High | 3x |
| Task 14: Model Comparison | 6–8 | +10 | Medium | 1.4x |

**Quick Wins Total:** 37–47 hours → **+103–129 points (capped)**  
**Expected ROI:** ~2.4x average

### Major Projects (Higher Effort, Long-Term Value)

| Task | Hours | Gain | ROI | Strategic Value |
|------|-------|------|-----|-----------------|
| Task 2: Tensor Products | 12–15 | +25–30 | 2x | Enables 2D/3D modeling |
| Task 5: Cubic Spline | 10–12 | +15–20 | 1.5x | Fast alternative basis |
| Task 7: B-Spline | 12–15 | +15–20 | 1.3x | Flexible basis |
| Task 8: P-Spline | 6–8 | +10–15 | 1.5x | Eilers-Marx standard |
| Task 10: Confidence Intervals | 10–12 | +18–25 | 1.8x | Better inference |
| Task 11: GCV Completion | 6–8 | +15 | 2x | Most popular criterion |
| Task 13: REML Fixing | 8–10 | +10–15 | 1.3x | Bayesian angle |

**Major Projects Total:** 64–80 hours → **+108–135 points (capped)**  
**Expected ROI:** ~1.6x average

---

## PART V: RECOMMENDED EXECUTION STRATEGY

### Option A: Fastest Path to 98% (21 days, High Focus)

**Recommended** for achieving goal within timeline.

1. **Days 1–2:** Phase 1 (Task 1, PIRLS) + Task 6 (sp=)
2. **Days 3–6:** Phase 2A (Tasks 3, 4 parallel) + Task 5 start (cubic)
3. **Days 7–14:** Phase 2B (Tasks 5, 7, 8, maybe 2) + Task 9-10 parallel
4. **Days 15–21:** Phase 3 (Tasks 9, 10, 14) + Phase 4 start (Tasks 11-13)

**Effort:** 114–144 hours (~7 hrs/day for 21 days)

**Expected Outcome:** 96–98% parity

**Risks:**
- High context switching (need deep focus blocks)
- Limited time for thorough testing
- May discover integration issues requiring backfitting

---

### Option B: Thorough & Parallel (25 days, Maximum Parallelization)

1. **Days 1–3:** Phase 1 (Task 1) → triggers Phase 2A, 2B, start design
2. **Days 3–8:** Phase 2A full parallel (Tasks 3, 4, 6) + Phase 2B design (Tasks 2, 5, 7)
3. **Days 9–17:** Phase 2B implementation (Tasks 2, 5, 7, 8 sequential, ~1–2/week)
4. **Days 15–21:** Phase 3 parallel (Tasks 9, 10, 14)
5. **Days 22–25:** Phase 4 (Tasks 11, 12, 13)

**Effort:** 114–144 hours (~6 hrs/day for 25 days)

**Expected Outcome:** 97–98% parity with higher confidence

**Advantages:**
- Better parallelization (fewer bottlenecks)
- More thorough testing per component
- Fewer integration surprises
- Stronger code quality

---

### Option C: Conservative (35+ days, Maximum Quality)

Include Phase 5 optional enhancements + thorough documentation.

**Effort:** 164–217 hours (~5–6 hrs/day for 35+ days)

**Expected Outcome:** 98%+ parity with polish, documentation, examples

---

## PART VI: CRITICAL BLOCKERS & RISKS

### Known Blockers

1. **PIRLS Numerical Stability (Task 1)**
   - **Severity:** 🔴 CRITICAL
   - **Impact:** Blocks all non-Gaussian fitting
   - **Mitigation:** Implement early (Days 1–2)
   - **Backup:** Gaussian-only for immediate delivery, then fix

2. **JAX Integration in PIRLS**
   - **Severity:** 🟠 HIGH
   - **Impact:** GPU acceleration disabled for non-Gaussian
   - **Mitigation:** Ensure JAX updates work with safeguards
   - **Backup:** NumPy fallback OK (slower but correct)

3. **Basis Dimension Interaction with MAGIC**
   - **Severity:** 🟠 HIGH
   - **Impact:** Too many degrees of freedom → over-fitting
   - **Mitigation:** Validate cubic/bspline/pspline with mgcv across k values
   - **Backup:** Conservative default k values with warning

### Identified Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Tensor product Kronecker memory explosion | Low (15%) | High | Implement sparse matrix support |
| Cyclic basis boundary conditions complex | Medium (40%) | Medium | Defer Task 15 if needed |
| REML optimization convergence issues persist | Medium (35%) | Low | Document limitations, use GCV default |
| Weights interaction with penalties surprises | Low (20%) | Medium | Extensive unit testing before integration |
| By-variable dimension explosion (many levels) | Low (15%) | High | Implement factor grouping/binning |

---

## PART VII: VALIDATION & TESTING STRATEGY

### Testing Infrastructure Setup (First 1–2 days)

```python
# Create comparison test suite
class TestPyMGCVvsMGCV:
    """Unified testing harness comparing pymgcv to R mgcv output"""
    
    TOLERANCE = 1e-4  # Relative tolerance for most metrics
    
    def test_family_coef(self, family, formula, data_r, tolerance=1e-6):
        """Compare coefficients for each family"""
        
    def test_edf(self, model_py, model_r, tolerance=0.01):
        """EDF should agree to 0.01"""
        
    def test_smoothing_parameters(self, model_py, model_r, tolerance=0.05):
        """λ should agree within 5%"""
        
    def test_predictions(self, model_py, model_r, test_data, tolerance=1e-4):
        """Predictions should match numerically"""
```

### Per-Task Testing

Each task includes:
1. **Unit tests:** Component isolation (basis matrix, penalty matrix, etc.)
2. **Integration tests:** With solver (PIRLS, MAGIC)
3. **R Comparison tests:** Output comparison to mgcv
4. **Edge cases:** Small n, singular matrices, boundary conditions

### Phased Validation Checkpoints

```
✓ After Task 1:    All families show <5% fitting error vs mgcv
✓ After Task 3,4:  by-variables, weights produce expected design matrix shapes
✓ After Task 5,7:  Cubic/B-spline fit quality similar to TPRS
✓ After Task 2:    Tensor products 2D plots match mgcv visually
✓ After Task 9:    gam.check() flags match mgcv diagnostics
✓ After Task 10:   Confidence bands contain 95% of bootstrap samples
✓ After Task 11:   GCV values ~same as MAGIC for same data
✓ Final:           Full comparison suite: 47 tests, all pass ✓
```

---

## PART VIII: DOCUMENTATION & DELIVERY

### Documentation Needed Per Task

| Category | Tasks | Deliverable | Effort |
|----------|-------|-------------|--------|
| **API Docs** | 1-6 | Updated docstrings, type hints | 3 hrs |
| **User Guide** | 2-4 | "How to use tensor products", "by-variables", etc. | 4 hrs |
| **Examples** | 1-6 | Jupyter notebooks demonstrating each feature | 8 hrs |
| **Theory** | 1-2, 7-8 | Mathematical background in IMPLEMENTATION_ROADMAP_THEORY.md | 6 hrs |
| **Troubleshooting** | 1, 13 | Known issues, convergence tips | 2 hrs |

**Total Documentation Effort:** ~23 hours

---

## FINAL RECOMMENDATIONS

### To Achieve 98% Parity in 3 Weeks (21 Days)

**Commit to:**
1. ✅ Task 1: PIRLS Stability (non-negotiable blocker)
2. ✅ Task 3: By-Variables (critical feature)
3. ✅ Task 4: Weights (critical feature)
4. ✅ Task 5: Cubic Spline (complete smooth bases)
5. ✅ Task 7: B-Spline (complete smooth bases)
6. ✅ Task 9: gam.check() (critical diagnostics)
7. ✅ Task 10: Confidence Intervals (improve inference)
8. ✅ Task 11: GCV (optimization)

**Nice-to-Have (if time permits):**
- Task 2: Tensor Products (+25–30 points, 12–15 hrs)
- Task 8: P-Spline (+10–15 points, 6–8 hrs)
- Task 13: REML (+10–15 points, 8–10 hrs)

**Expected Outcome:** 94–98% parity, 114–140 hours effort, 21–25 days wallclock

### Parallel Workstreams (Maximize Efficiency)

**Stream A (Solver):** Task 1 alone (Days 1–2)

**Stream B (Specification):** Tasks 3, 4, 6 parallel (Days 3–7)

**Stream C (Bases):** Tasks 5, 7, 8, (2) sequential (Days 3–17)
- Start Day 3, with design phase during Stream A

**Stream D (Diagnostics & Optimization):** Tasks 9, 10, 11, 12, 14 parallel (Days 15–21)
- Unblock after Task 1 complete

**Total Parallelizable Effort:** ~40% time savings vs sequential

---

## APPENDIX: SCORECARD PROJECTION

### Before Implementation

```
Families:       64.3/100
Smooth Bases:   34.7/100
Optimization:   45.0/100
Inference:      50.0/100
Diagnostics:    43.8/100
Specification:  35.0/100
─────────────────────────
OVERALL:        46.7/100
```

### After Completion (Recommended 8 Tasks)

```
Families:       79.3/100  (+15.0) — Task 1 (PIRLS)
Smooth Bases:   60–65/100 (+25–30) — Tasks 5,7, partial Task 2
Optimization:   65–70/100 (+20–25) — Task 11 (GCV) + partial 13
Inference:      70–75/100 (+20–25) — Tasks 9,10
Diagnostics:    68–73/100 (+24–30) — Task 9 (gam.check)
Specification:  60–65/100 (+25–30) — Tasks 3,4,6
─────────────────────────────────────────────────
OVERALL:        94–98/100  (+47–51 points) ✅✅✅
```

---

**End of Roadmap**

---

### How to Use This Document

1. **For Project Planning:** Use Part II (sequencing) and Part III (effort estimates)
2. **For Prioritization:** Reference Part IV (quick wins vs major projects)
3. **For Implementation:** Follow Part I (detailed task specs) in order
4. **For Validation:** Use Part VII (testing strategy)
5. **For Risk Management:** Reference Part VI (blockers and risks)

**Next Steps:**
1. Confirm timeline/effort allocation with team
2. Set up version control branches for parallel streams
3. Create GitHub Issues from each task in Part I
4. Begin with Task 1 (PIRLS Stability) immediately
5. Assign Tasks 3–6 to parallel team members once Task 1 is 50% complete

