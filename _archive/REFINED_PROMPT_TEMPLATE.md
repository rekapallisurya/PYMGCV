# REFINED PROMPT TEMPLATE FOR PYMGCV OPTIMIZATION

## Instructions

After you've compared PyMGCV output with R's mgcv output, use this template to create a refined prompt that focuses the next iteration of fixes.

---

## Template for Refined Prompt

```markdown
# PyMGCV Refinement Prompt - [Date]

## Comparison Results

I've compared PyMGCV with R's mgcv and here are the detailed results:

### Dataset Information
- Sample size: [n]
- Formula: [formula]
- Family: [family]
- Data generation seed: [seed]

### Numerical Comparison

#### Parametric Coefficients
| Coefficient | PyMGCV | R mgcv | Difference | Tolerance Status |
|-------------|--------|--------|-----------|------------------|
| Intercept | [val] | [val] | [diff] | [Pass/Fail ε<1e-6] |
| [Other] | [val] | [val] | [diff] | [Pass/Fail] |

#### Smooth Term Statistics
| Term | PyMGCV EDF | R mgcv EDF | Diff | Status |
|------|------------|-----------|------|--------|
| s(x) | [val] | [val] | [diff] | [Pass/Fail ε<0.01] |

#### Model Criteria
| Criterion | PyMGCV | R mgcv | Match? |
|-----------|--------|--------|--------|
| AIC | [val] | [val] | [Yes/No] |
| GCV | [val] | [val] | [Yes/No] |
| Deviance | [val] | [val] | [Yes/No] |

#### Smoothing Parameters
| Term | PyMGCV λ | R mgcv λ | Ratio |
|------|----------|----------|-------|
| s(x) | [val] | [val] | [ratio] |

#### Predictions
- First 5 predictions (PyMGCV): [vals]
- First 5 predictions (R mgcv): [vals]
- Max difference: [diff]
- Status: [Pass/Fail ε<1e-6]

### Issues Identified

#### Priority 1 (Breaks Equivalence)
1. **Issue**: [Description]
   - **Magnitude**: [Difference amount]
   - **Location**: [Component/function]
   - **Impact**: Affects [coefficients/EDF/AIC/etc]
   - **Suspected Cause**: [Analysis]
   - **Desired Fix**: [Solution approach]

#### Priority 2 (Affects Accuracy)
1. **Issue**: [Description]
   - **Impact**: [Effect on output]

#### Priority 3 (Code Quality)
1. **Issue**: [Description]

### Specific Code Locations Needing Attention

```
File: [path]
Function: [function_name]
Issue: [Detailed issue]
Expected behavior: [What mgcv does]
Current behavior: [What pymgcv does]
```

### Test Cases to Add/Fix

```python
def test_coefficient_match_mgcv():
    # Test that coefficients match R within 1e-6
    pass

def test_edf_match_mgcv():
    # Test that EDF matches R within 0.01
    pass
```

### Refactoring Tasks

- [ ] [Task 1]
- [ ] [Task 2]
- [ ] [Task 3]

### Estimated Impact

- **Complexity**: Low / Medium / High
- **Expected improvement**: [Brief description of expected gains]
- **Time estimate**: [Hours needed]

### Additional Context

[Any other relevant information, plots, data samples, etc.]

---

## Accompanying Files to Share

Please provide:

1. **R output (copy and paste)**: 
   ```r
   # Paste output from:
   # summary(fit)
   # coef(fit)
   # fit$edf
   # AIC(fit)
   # fit$gcv.ubre
   ```

2. **PyMGCV output (copy and paste)**:
   ```
   # From model.summary()
   ```

3. **Generated data (first 10 rows)**:
   ```
   # From data.head(10) as CSV or JSON
   ```

4. **Detailed comparison table** (as above)

---

## Example of Well-Formatted Refined Prompt

**[Good Example]**

```markdown
# PyMGCV Refinement Prompt - Session 2

## Dataset Information
- Sample size: 150
- Formula: y ~ s(x, k=10)
- Family: gaussian()
- Seed: 42

## Numerical Comparison

### Parametric Coefficients
| Coefficient | PyMGCV | R mgcv | Difference | Status |
|-------------|--------|--------|-----------|--------|
| Intercept | 0.021540 | 0.021541 | 1e-6 | ✓ PASS |

### EDF
| Term | PyMGCV | R mgcv | Difference | Status |
|------|--------|--------|-----------|--------|
| s(x) | 2.4512 | 2.4515 | 0.0003 | ✓ PASS |

### Model Criteria
| Criterion | PyMGCV | R mgcv | Status |
|-----------|--------|--------|--------|
| AIC | -234.567 | -234.567 | ✓ Match |
| GCV | 0.31245 | 0.31245 | ✓ Match |

## Issue: Smoothing parameter optimization

- **Magnitude**: λ_py = 0.01234, λ_R = 0.01236
- **Difference**: 0.016% (relative)
- **Impact**: Negligible on fit quality (GCV/AIC match)
- **Status**: Acceptable - likely due to different optimization convergence

## Conclusion

PyMGCV numerical equivalence achieved within target tolerance (1e-6 for coefficients, 0.01 for EDF, identical for AIC/GCV).

Next: Validate on additional test cases (Poisson, Gamma, Tweedie families).
```

```

---

## Quick Checklist for Your R Comparison

Before running R code:
- [ ] Data generated with identical seed (set.seed(42))
- [ ] Same formula syntax converted between languages
- [ ] Same family/link specified in both
- [ ] No data cleaning done that differs
- [ ] Both use same basis dimension (k=10, etc.)

While running R code:
- [ ] Save all output to text file
- [ ] Extract coefficients: `coef(fit)`
- [ ] Extract EDF: `fit$edf`
- [ ] Extract smoothing params: `fit$sp`
- [ ] Extract summary: `summary(fit)`
- [ ] Compute predictions: `predict(fit, newdata)`
- [ ] Check for any warnings/errors

Submitting results:
- [ ] R output (complete summary)
- [ ] PyMGCV output (complete summary)
- [ ] CSV with numerical comparisons
- [ ] First 20 rows of actual data used
- [ ] Any plots or visualizations if discrepancies found

---

## Expected Format for Comparison Table

Use this template for your comparison:

```
COEFFICIENT COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Param         │ PyMGCV Value  │ R Value       │ Difference │ Relative Error
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Intercept     │ 0.021540      │ 0.021541      │ -1.0e-6    │ 0.0046%
β₁            │ 0.123456      │ 0.123457      │ -1.0e-6    │ 0.0081%
β₂            │ -0.098765     │ -0.098766     │ 1.0e-6     │ 0.0101%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EDF COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Term         │ PyMGCV EDF │ R EDF    │ Difference │ Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
s(x)         │ 2.4512     │ 2.4515   │ -0.0003    │ PASS (Δ < 0.01)
s(z)         │ 3.1234     │ 3.1256   │ -0.0022    │ PASS (Δ < 0.01)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL CRITERIA COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Criterion    │ PyMGCV      │ R mgcv      │ Match?  │ Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AIC          │ -234.5678   │ -234.5678   │ Yes     │ ✓ PASS
GCV          │ 0.312456    │ 0.312456    │ Yes     │ ✓ PASS
Deviance     │ 45.62341    │ 45.62341    │ Yes     │ ✓ PASS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## When You're Ready

Once you've completed the R comparison:

1. **Copy this template**
2. **Fill in your specific numbers**
3. **Identify any issues** using the Priority levels
4. **Share the refined prompt**
5. **I'll implement targeted fixes** to achieve equivalence

Your detailed comparison will allow me to:
- Focus on specific numerical problems
- Prioritize high-impact fixes
- Validate solutions against known good values
- Achieve 1e-6 numerical equivalence systematically
```

---

## Additional Resources

### Running R Code

If you don't have R installed, consider:
- Online R environment: https://www.r-project.org/
- RStudio Cloud: https://rstudio.cloud/
- Google Colab with R kernel

### Documentation Links

- PyMGCV docs: (to be created)
- R mgcv docs: https://www.maths.ed.ac.uk/~swood/mgcv/
- mgcv R package: `?mgcv::gam `

### Common R Commands for Comparison

```r
# Install packages
install.packages("mgcv")
library(mgcv)

# Load data
data <- read.csv("gam_data.csv")

# Fit model
fit <- gam(y ~ s(x, k=10), family=gaussian())

# View summary
summary(fit)

# Extract components
coef(fit)                    # Coefficients
fit$coefficients             # Same
fit$edf                      # EDF per term
fit$sp                       # Smoothing parameters
AIC(fit)                     # AIC
fit$gcv.ubre                 # GCV score
deviance(fit)                # Deviance

# Predictions
pred <- predict(fit, newdata=data.frame(x=seq(0,1,0.1)))

# Standard errors
se <- predict(fit, newdata=..., se.fit=TRUE)$se.fit

# Save results
results <- list(
    coef = coef(fit),
    edf = fit$edf,
    sp = fit$sp,
    aic = AIC(fit),
    gcv = fit$gcv.ubre,
    predictions = pred
)
saveRDS(results, "mgcv_results.rds")
```

---

**Ready to share your R comparison results?**

Use this template to structure your findings and I'll implement the necessary refinements to achieve 1e-6 numerical equivalence with R's mgcv!
