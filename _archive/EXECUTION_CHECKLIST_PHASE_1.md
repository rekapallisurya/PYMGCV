# Phase 1 Execution Checklist: Quick Wins (47% → 62% Parity)

## Timeline: 2 Days | Effort: 11 Hours | Expected Gain: +14.7 Points

---

## Overview

Phase 1 focuses on **quick, high-impact fixes** that stabilize the core GAM functions:

1. **PIRLS Stability** (4h) - Fix numerical convergence issues
2. **Weights Integration** (4h) - Enable weight support in likelihood
3. **Summary Formatting** (2h) - Better output formatting
4. **Offset Edge Cases** (1h) - Handle offset issues

---

## Task 1.1: Fix PIRLS Stability (4 hours)

### Current State
```python
# Location: pymgcv/optimizer/pirls.py
# Status: ❌ Converges poorly for some family/data combinations
# Evidence: Binomial fits fail on 15% of test datasets
```

### What's Broken
- No line search (full Newton steps can overshoot)
- Convergence check only on change in eta, not deviance
- No handling of infinite/NaN iterates
- Step-halvings not implemented

### Implementation Checklist

#### 1.1.1 Add Line Search (1.5h)
```python
# File: pymgcv/optimizer/pirls.py
# Add to PIRLS class:

def _line_search(self, eta_old, eta_new, y, weights, family):
    """Perform backtracking line search for step size."""
    step_size = 1.0
    dev_old = family.deviance(y, eta_old, weights)
    
    for _ in range(10):  # Max 10 line search iterations
        eta_trial = eta_old + step_size * (eta_new - eta_old)
        dev_trial = family.deviance(y, eta_trial, weights)
        
        if dev_trial < dev_old * 0.99:  # Accept if 1% improvement
            return eta_trial, step_size
        
        step_size *= 0.5  # Halve step
    
    # Fallback: return original if no improvement
    return eta_old, 0
```

**Testing:**
```python
# test_pirls_line_search.py
def test_line_search_prevents_divergence():
    """Line search should prevent overshoot"""
    # Create difficult case: separation in binomial
    y = np.array([0]*10 + [1]*10)
    X = np.array([[-10, 1], [10, 1]] * 10)
    
    pirls = PIRLS()
    eta_converged = pirls.fit(X, y, family='binomial')
    
    assert np.all(np.isfinite(eta_converged))
    assert np.all(np.abs(eta_converged) < 100)  # Bounded
```

#### 1.1.2 Improve Convergence Check (1h)
```python
# File: pymgcv/optimizer/pirls.py
# Replace convergence logic:

def _has_converged(self, dev_old, dev_new, eta_diff, tol=1e-8):
    """Check convergence via multiple criteria."""
    criteria = [
        ("deviance", abs(dev_old - dev_new) < tol),
        ("eta", np.max(np.abs(eta_diff)) < tol),
        ("relative", abs(dev_old - dev_new) / (abs(dev_old) + 1e-10) < 1e-6),
    ]
    
    # All criteria must pass
    return all(c[1] for c in criteria)
```

#### 1.1.3 Add NaN Handling (1.5h)
```python
# File: pymgcv/optimizer/pirls.py
# Add to fit() method:

for iteration in range(max_iterations):
    # ... standard PIRLS update ...
    
    # 🔴 NEW: Check for NaN/Inf
    if not np.all(np.isfinite(eta_new)):
        print(f"⚠️  Iteration {iteration}: Non-finite eta detected")
        # Backtrack to best previous eta
        if iteration > 0:
            eta_new = eta_old
            step_size = 0.5
        else:
            raise ValueError("Initial eta is non-finite")
    
    # Continue with line search...
```

#### 1.1.4 Validation
- [ ] **Test 1:** `test_pirls_gaussian_convergence()` - Basic case
- [ ] **Test 2:** `test_pirls_poisson_count_data()` - Count data
- [ ] **Test 3:** `test_pirls_binomial_separation()` - Hard case (0/1 separation)
- [ ] **Test 4:** `test_pirls_weights_heteroscedasticity()` - Weights
- [ ] **Test 5:** Compare vs mgcv on 10 datasets - All within ±1e-4

---

## Task 1.2: Integrate Weights (4 hours)

### Current State
```python
# Location: pymgcv/optimizer/pirls.py, pymgcv/api/gam.py
# Status: ⚠️ Weights partially working, not fully in PIRLS
```

### What's Needed
- Weights in likelihood calculation
- Weights in Fisher information matrix
- Weights in variance calculations
- Offset & weight interaction

### Implementation Checklist

#### 1.2.1 Update PIRLS Likelihood (1h)
```python
# File: pymgcv/optimizer/pirls.py
# Replace deviance calculation:

def _compute_weighted_deviance(self, y, mu, weights):
    """Compute deviance with weights."""
    # Standard: deviance = 2 * sum((y - mu)² / var(mu))
    # Weighted: deviance = 2 * sum(weights * (y - mu)² / var(mu))
    
    variance = self.family.variance(mu)
    weighted_dev = 2.0 * np.sum(
        weights * ((y - mu) ** 2) / (variance + 1e-10)
    )
    return weighted_dev
```

#### 1.2.2 Update Fisher Information (1.5h)
```python
# File: pymgcv/optimizer/pirls.py
# Modify W matrix construction:

def _compute_working_weights(self, mu, weights):
    """Weighted working weights matrix."""
    # Standard: W = diag(d*mu/d*eta)² / var(mu)
    # Weighted: W = diag(weights * (d*mu/d*eta)² / var(mu))
    
    dmu_deta = self.family.link.deriv(mu)
    variance = self.family.variance(mu)
    
    # Element-wise: w_i = weights_i * (dμ_i/dη_i)² / var(μ_i)
    w_matrix = weights * (dmu_deta ** 2) / (variance + 1e-10)
    
    return np.diag(w_matrix)
```

#### 1.2.3 Update Variance Calculation (1h)
```python
# File: pymgcv/api/gam.py (after model fitting)
# Adjust SE calculation for weights:

def compute_standard_errors(self, X, weights):
    """Standard errors with weights."""
    if weights is None:
        weights = np.ones(len(X))
    
    # Weighted XtX: X' * W * X where W = diag(weights) * Fisher
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    
    # Pseudo-inverse for stability
    try:
        cov_matrix = np.linalg.inv(XtWX + 1e-8 * np.eye(XtWX.shape[0]))
    except:
        cov_matrix = np.linalg.pinv(XtWX)
    
    se = np.sqrt(np.diag(cov_matrix))
    return se
```

#### 1.2.4 Validation
- [ ] **Test 1:** `test_weights_unweighted_equivalence()` - weights=1 = no weights
- [ ] **Test 2:** `test_weights_heteroscedasticity()` - Known variance
- [ ] **Test 3:** `test_weights_gaussian()` - Gaussian family
- [ ] **Test 4:** `test_weights_poisson()` - Poisson family
- [ ] **Test 5:** Compare vs mgcv on weighted data

---

## Task 1.3: Summary Formatting (2 hours)

### Current State
```python
# Location: pymgcv/api/summary.py
# Status: ⚠️ Basic output, missing tables and statistics
```

### What's Needed
- Full parametric coefficient table
- Smooth term significance table
- Model statistics (deviance explained, AIC, GCV)
- Proper formatting with significance markers (\*, \*\*, \*\*\*)

### Implementation

#### 1.3.1 Parametric Coefficients Table (0.5h)
```python
# File: pymgcv/api/summary.py
# Update print_parametric_coefficients():

def format_parametric_section(self):
    """Format parametric coefficients like mgcv."""
    lines = []
    lines.append("Parametric coefficients:")
    lines.append("-" * 75)
    lines.append(f"{'':20} {'Estimate':>12} {'Std. Error':>12} {'t value':>10} {'Pr(>|t|)':>12}")
    lines.append("-" * 75)
    
    for i, (name, (coef, se)) in enumerate(zip(self.coef_names, zip(self.beta, self.se))):
        if se > 0:
            t_val = coef / se
            p_val = 2 * (1 - stats.t.cdf(abs(t_val), self.n - self.p))
        else:
            t_val = np.nan
            p_val = np.nan
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        
        lines.append(
            f"{name:20} {coef:12.6f} {se:12.6f} {t_val:10.4f} {p_val:12.4e} {sig}"
        )
    
    return "\n".join(lines)
```

#### 1.3.2 Smooth Terms Table (1h)
```python
# File: pymgcv/api/summary.py
# Add format_smooth_section():

def format_smooth_section(self):
    """Format smooth term significance."""
    lines = []
    lines.append("Approximate significance of smooth terms:")
    lines.append("-" * 75)
    lines.append(f"{'':20} {'edf':>8} {'Ref.df':>8} {'F':>10} {'p-value':>12}")
    lines.append("-" * 75)
    
    for smooth_name, edf in zip(self.smooth_names, self.edf_values):
        # Compute F-statistic (simplified)
        f_stat = self._compute_f_statistic(smooth_name)
        p_val = self._compute_p_value(f_stat, edf)
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        
        lines.append(
            f"{smooth_name:20} {edf:8.2f} {edf+2:8.2f} {f_stat:10.2f} {p_val:12.4e} {sig}"
        )
    
    return "\n".join(lines)
```

#### 1.3.3 Model Statistics (0.5h)
```python
# File: pymgcv/api/summary.py
# Add format_statistics():

def format_statistics(self):
    """Format final model statistics."""
    dev_expl = self.deviance_explained_pct
    aic = self.aic
    gcv = self.gcv_score
    
    return f"""
R-sq.(adj) = {self.r2_adj:.3f}
Deviance explained = {dev_expl:.1f}%
GCV score: {self.gcv_score:.6f}
AIC: {aic:.2f}

Method: {self.method} | Optimizer: {self.optimizer}
Convergence: {'Yes' if self.converged else 'No'} ({self.iterations} iterations)
"""
```

#### 1.3.4 Validation
- [ ] **Test 1:** `test_summary_format_matches_mgcv()` - Format comparison
- [ ] **Test 2:** `test_summary_gaussian()` - Gaussian output
- [ ] **Test 3:** `test_summary_poisson()` - Poisson output
- [ ] **Test 4:** `test_summary_binomial()` - Binomial output
- [ ] **Visual:** Compare with actual R mgcv output

---

## Task 1.4: Offset Edge Cases (1 hour)

### Current State
```python
# Status: ⚠️ Offsets work for basic cases, fail on edge cases
```

### Fixes Needed

#### 1.4.1 Handle Zero/Infinite Offsets
```python
# File: pymgcv/optimizer/pirls.py
# Add validation:

def validate_offset(self, offset, n):
    """Validate offset parameter."""
    if offset is None:
        return np.zeros(n)
    
    offset = np.asarray(offset)
    
    # Check dimensions
    if len(offset) != n:
        raise ValueError(f"Offset length {len(offset)} != n={n}")
    
    # Handle infinite offsets
    if not np.all(np.isfinite(offset)):
        print(f"⚠️  Offset contains {np.sum(~np.isfinite(offset))} non-finite values")
        offset = np.where(np.isfinite(offset), offset, 0)
    
    return offset
```

#### 1.4.2 Offset in Link Scale Check
```python
# File: pymgcv/optimizer/pirls.py
# Add to fit():

# Ensure eta_init = X @ beta + offset is in valid range
eta_init = X @ beta_init + offset
mu_init = self.family.link.inv(eta_init)

# Validate initial mu
if np.any(mu_init <= 0) and self.family.name in ['poisson', 'gamma']:
    print(f"⚠️  Initial mu has non-positive values for {self.family.name}")
    mu_init = np.maximum(mu_init, 1e-5)
```

#### 1.4.3 Validation
- [ ] **Test 1:** `test_offset_zero()` - Offset=0 works
- [ ] **Test 2:** `test_offset_constant()` - Constant offset works
- [ ] **Test 3:** `test_offset_variable()` - Variable offset works
- [ ] **Test 4:** `test_offset_edge_negative()` - Can handle large negative offsets
- [ ] **Test 5:** `test_offset_infinity_handling()` - Graceful handling

---

## Integration & Testing

### Day 1 Afternoon: Integration Testing

```bash
# Run all Phase 1 tests
cd c:\Users\surya\Downloads\pymgcv

pytest tests/test_phases_1_2.py -v
pytest tests/test_integration.py -v
pytest tests/test_validation_mgcv.py -v

# Run new Phase 1 tests
pytest tests/phase_1_tests.py -v --cov=pymgcv/optimizer/pirls
```

### Day 2 Morning: Benchmark vs R

```r
# Run in R (save as benchmark.R)
library(mgcv)

# Test 1: Gaussian (simple)
y1 <- rnorm(100)
x1 <- seq(0,1,len=100)
m1 <- gam(y1 ~ s(x1))
coef(m1)

# Test 2: Poisson (count data)
y2 <- rpois(100, 5)
m2 <- gam(y2 ~ s(x1), family=poisson())
coef(m2)

# Test 3: Binomial (weights)
y3 <- rbinom(100, 1, 0.5)
weights <- rep(c(1, 2), 50)
m3 <- gam(y3 ~ s(x1), family=binomial(), weights=weights)
coef(m3)
```

```python
# Compare in Python
import numpy as np
import pandas as pd
from pymgcv.api import GAM

# Test 1: Gaussian
y1 = np.random.normal(0, 1, 100)
x1 = np.linspace(0, 1, 100)
m1 = GAM()
m1.fit(pd.DataFrame({'x': x1, 'y': y1}), 'y ~ s(x)', family='gaussian')
print("Python Gaussian:", m1.beta)
print("R Gaussian: [from R output]")
print("Difference:", np.abs(m1.beta - r_coef1))

# Compare similarly for Poisson, Binomial
```

---

## Phase 1 Success Criteria

✅ **All 5 Tasks Complete When:**

1. **PIRLS Stability**
   - ✅ All PIRLS tests pass
   - ✅ No convergence failures on 50+ datasets
   - ✅ Line search reduces iterations by 10-20%

2. **Weights Integration**
   - ✅ Weighted vs unweighted equivalence test passes
   - ✅ Heteroscedastic data fits correctly
   - ✅ Variance calculation with weights matches theory

3. **Summary Formatting**
   - ✅ Output visually matches mgcv
   - ✅ All significance markers present
   - ✅ Statistics calculations correct

4. **Offset Edge Cases**
   - ✅ No errors on edge case data
   - ✅ Results match mgcv ±1e-4

5. **Overall**
   - ✅ Parity score increased to 62% (from 47%)
   - ✅ All tests pass with `pytest -v`
   - ✅ Code coverage > 85% for touched modules

---

## Commit & Next Steps

### End of Phase 1:
```bash
git add -A
git commit -m "Phase 1: Fix PIRLS stability, implement weights, improve summary (47% → 62% parity)"
git push origin main
```

### Ready for Phase 2A?
Review `EXECUTION_CHECKLIST_PHASE_2A.md` to continue improving families & optimization.

---

**Estimated Completion: Day 2, 5pm** ⏰
