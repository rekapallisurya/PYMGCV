# PyMGCV Roadmap: Task Breakdown & Implementation Details

**Generated:** March 16, 2026  
**Scope:** Detailed specifications for each of 16 core tasks  
**Reference:** See IMPLEMENTATION_ROADMAP_TO_98_PARITY.md for overall strategy

---

## TASK SPECIFICATIONS (Quick Reference)

### ⭐ TIER 1: BLOCKING FIXES (Must do first - Days 1-7)

---

#### TASK 1: Fix PIRLS Numerical Stability for Non-Gaussian Families

**Priority:** 🔴 CRITICAL (blocks all other families)

**Current State:**
- File: `pymgcv/optimizer/pirls.py`
- Status: Works for Gaussian, Poisson; crashes on Binomial, NB, IG
- Root cause: Lines 129-132 produce NaN/Inf in weights and working variables

**What Needs to Happen:**
1. Clone `pymgcv/optimizer/pirls.py` → backup first
2. In weight calculation (~line 129):
   ```python
   # BEFORE: w = (dmu_deta**2) / var_mu
   # AFTER:
   dmu_deta_safe = np.clip(dmu_deta, 1e-10, np.inf)
   var_mu_safe = np.clip(var_mu, 1e-10, np.inf)
   w = np.clip((dmu_deta_safe**2) / var_mu_safe, 1e-10, 1e10)
   ```

3. In working variable (~line 132):
   ```python
   # Add check for zero derivative
   with np.errstate(divide='ignore', invalid='ignore'):
       z = eta + (y - mu) / dmu_deta_safe
       z = np.where(np.isfinite(z), z, eta)  # fallback to eta if non-finite
   ```

4. Add damping/step-halving (~15-20 new lines after iteration):
   ```python
   # After updating beta, check for divergence
   loss_new = compute_loss(...)
   if loss_new > loss_old:
       # Reduce step size
       beta = 0.5 * beta + 0.5 * beta_old
       # Retry
   ```

5. Add convergence safeguards before iteration:
   ```python
   if np.any(~np.isfinite(beta)) or np.any(~np.isfinite(w)):
       warnings.warn("PIRLS divergence detected")
       return beta_last_good, False  # Return last valid iterate
   ```

**Key Code Locations:**
- Weight calc: Line ~129
- Working var: Line ~132
- Solve step: Line ~145-160
- Loop termination: Line ~200+

**Testing:**
```python
def test_pirls_binomial():
    # Should fit without NaN
    from pymgcv.api.gam import GAM
    x = np.random.randn(50)
    y = np.random.binomial(1, 0.5, 50)
    m = GAM('y ~ s(x)', family='binomial').fit()
    assert np.all(np.isfinite(m.beta))
    assert m.fitted
```

**Deliverables:**
- Modified `pymgcv/optimizer/pirls.py` (safeguards + damping)
- Updated unit tests (5+ test cases)
- Before/after performance metrics

**Success Criteria:**
- ✅ Binomial GAM fitting converges (no NaN)
- ✅ NB/IG fitting also converges
- ✅ Gaussian/Poisson still work identically
- ✅ Coefficients match R mgcv to 1e-4 tolerance

**Estimated Time:** 4–5 hours

**ROI:** Unlocks 3 distributions + enables all downstream tasks

---

#### TASK 2: Implement Tensor Product Smooths (te/ti/t2)

**Priority:** 🟠 HIGH (essential for 2D/3D smoothing)

**Current State:**
- File: Does not exist (need to create `pymgcv/smooth/tensor_product.py`)
- Status: Score 0/100
- Impact: ~40% of real-world GAM applications require 2D+ smooths

**Files to Create/Modify:**
1. **CREATE:** `pymgcv/smooth/tensor_product.py` (~400-500 lines)
2. **MODIFY:** `pymgcv/utils/model_matrix.py` (add tensor product parsing ~50 lines)
3. **MODIFY:** `pymgcv/penalties/penalty_matrix.py` (add Kronecker logic ~80 lines)
4. **ADD TESTS:** `tests/test_tensor_products.py` (~200 lines)

**Implementation Steps:**

**Step 1: Parser Extension (~40 lines, in model_matrix.py)**
```python
def parse_tensor_smooth(formula_term):
    """Convert 'te(x, y, k=c(10,10))' → tensor spec"""
    # Extract variables: [x, y]
    # Extract basis types: default bs="tp" per dimension
    # Extract k values: [10, 10]
    # Return: {'type': 'tensor', 'vars': [x, y], 'basis': ['tp', 'tp'], 'k': [10, 10]}
    
def parse_tensor_interaction(formula_term):
    """Convert 'ti(x, y)' → tensor interaction spec"""
    # Same as te() but marks as "interaction only"
    # Margins are removed automatically
```

**Step 2: Univariate Margin Extraction (~120-150 lines, new file tensor_product.py)**
```python
class TensorProductBasis:
    def __init__(self, data, var_names, basis_types, k_values):
        # Store margins: one basis per variable
        self.margins = []
        for var, bs_type, k in zip(var_names, basis_types, k_values):
            basis = smooth_factory(data[var], bs_type, k)
            self.margins.append(basis)
    
    def get_margin_basis(self, margin_idx, data_col):
        """Get basis matrix for margin i"""
        return self.margins[margin_idx].basis_matrix(data_col)
    
    def get_margin_penalty(self, margin_idx):
        """Get penalty matrix for margin i"""
        return self.margins[margin_idx].penalty_matrix()
```

**Step 3: Full Tensor Assembly (~100-120 lines, in tensor_product.py)**
```python
def assemble_tensor_basis(margins):
    """
    Build full tensor product basis via Kronecker product
    
    For 2D: X = X₁ ⊗ X₂ (shape: n × (p₁*p₂))
    For 3D: X = X₁ ⊗ X₂ ⊗ X₃
    """
    X = margins[0].X  # Start with first margin
    for margin in margins[1:]:
        X = np.kron(X, margin.X)
    return X

def assemble_tensor_penalty(penalties):
    """
    Build full tensor penalty via Kronecker sum
    
    For 2D: P = P₁ ⊗ I₂ + I₁ ⊗ P₂
    For 3D: P = P₁⊗I₂⊗I₃ + I₁⊗P₂⊗I₃ + I₁⊗I₂⊗P₃
    """
    nterms = len(penalties)
    d = [pen.shape[0] for pen in penalties]  # dimensions
    
    P_sum = None
    for i in range(nterms):
        # Build i-th Kronecker term
        terms = []
        for j in range(nterms):
            if i == j:
                terms.append(penalties[j])
            else:
                terms.append(np.eye(d[j]))
        P_i = kronecker_sum(terms)
        
        if P_sum is None:
            P_sum = P_i
        else:
            P_sum = P_sum + P_i
    
    return P_sum
```

**Step 4: Tensor-of-Contrasts (ti) (~80-100 lines)**
```python
def assemble_tensor_interaction(margins):
    """
    Build tensor interaction (removes main effects)
    
    Each margin decomposed as: X = 1*μ' + (X - 1*μ')
    Interaction: (X₁ - 1*μ₁') ⊗ (X₂ - 1*μ₂')
    """
    de_meaned = []
    for margin in margins:
        X_margin = margin.X
        colmeans = X_margin.mean(axis=0)
        X_demeaned = X_margin - colmeans[np.newaxis, :]
        de_meaned.append(X_demeaned)
    
    X_int = de_meaned[0]
    for X_d in de_meaned[1:]:
        X_int = np.kron(X_int, X_d)
    
    return X_int
```

**Step 5: Integration with ModelMatrix (~80 lines)**
```python
# In pymgcv/utils/model_matrix.py, update assemble():

class ModelMatrix:
    def assemble(self, ...):
        # ... existing code ...
        
        for smooth_spec in smooth_specs:
            if smooth_spec['type'] == 'tensor':
                X_tensor = TensorProductBasis(...).basis_matrix()
                P_tensor = TensorProductBasis(...).penalty_matrix()
                
                design_matrix.append(X_tensor)
                penalties[smooth_name] = P_tensor
            
            elif smooth_spec['type'] == 'tensor_interaction':  # ti()
                X_int = TensorInteractionBasis(...).basis_matrix()
                # Same penalty as tensor
                
                design_matrix.append(X_int)
```

**Testing Strategy:**
```python
# tests/test_tensor_products.py

def test_tensor_product_basis_shape():
    X1 = np.random.randn(100, 10)
    X2 = np.random.randn(100, 8)
    X_tensor = np.kron(X1, X2)
    assert X_tensor.shape == (100, 80)

def test_tensor_product_fit():
    # 2D surface fit
    x1 = np.linspace(0, 1, 20)
    x2 = np.linspace(0, 1, 20)
    X1, X2 = np.meshgrid(x1, x2)
    y = np.sin(X1) * np.cos(X2) + noise
    
    model = GAM('y ~ te(x1, x2)').fit()
    assert model.fitted
    # Visual: smooth should show clear 2D pattern

def test_tensor_vs_mgcv():
    # Compare coefficients, EDF, GCV with R output
    # Tolerance: coefficients ±1e-4, EDF ±0.5
```

**Key Challenges:**
- Kronecker product memory scaling: O(p₁*p₂) for 2D (can be huge)
  - Mitigation: Use sparse or structured matrix representations if needed
- Penalty matrix conditioning: P is block-structured, ensure numerical stability
  - Mitigation: Eigen-truncation + regularization

**Deliverables:**
- `pymgcv/smooth/tensor_product.py` (full implementation)
- Updates to `model_matrix.py` and `penalty_matrix.py`
- Unit tests + R comparison tests
- Examples: 2D sine surface, contour plot

**Success Criteria:**
- ✅ 2D tensor product fits converge (no NaN)
- ✅ Coefficients match R mgcv to 1e-4
- ✅ EDF values agree within 0.5
- ✅ Visual comparison: contour plots look similar

**Estimated Time:** 12–15 hours

**ROI:** +25–30 points on Smooth Bases (single most impactful feature)

**Dependencies:** Task 1 (working solver)

---

#### TASK 3: Implement by-Variable Support (Varying-Coefficient Models)

**Priority:** 🔴 CRITICAL (essential feature for practical models)

**Current State:**
- File: `pymgcv/utils/formula_parser.py` (needs extension)
- Status: Score 0/100
- Impact: Many real models need group-specific effects

**Files to Modify:**
1. **MODIFY:** `pymgcv/utils/formula_parser.py` (~60 lines to add by parsing)
2. **MODIFY:** `pymgcv/utils/model_matrix.py` (~150 lines to expand basis)
3. **MODIFY:** `pymgcv/penalties/penalty_matrix.py` (~80 lines for by-penalty)
4. **ADD TESTS:** `tests/test_by_variables.py` (~150 lines)

**Implementation Steps:**

**Step 1: Formula Parsing (~40-60 lines)**
```python
# In formula_parser.py

class SmootherSpecification:
    # Add new field:
    by_variable: Optional[str] = None  # e.g., "group"
    by_continuous: bool = False  # continuous vs factor
    
def parse_smooth_spec(smooth_term_str):
    """Parse 's(x, by=group)' or 's(x, by=weight)'"""
    # Extract: base=(x), by_var=(group)
    # Determine type: factor or continuous
    # Return smooth spec with by_variable field
```

**Step 2: Design Matrix Expansion (~100-150 lines)**
```python
# In model_matrix.py

def expand_basis_with_by(X_basis, by_data, by_type):
    """
    Expand basis matrix for by-variable
    
    by_type='factor' with k levels:
        Input:  X_basis shape (n, p)
        Output: shape (n, k*p) with padding
        X_expanded[i, :] = [X[i]*I(by[i]==level_1), X[i]*I(by[i]==level_2), ...]
    
    by_type='continuous':
        Input:  X_basis shape (n, p)
        Output: shape (n, p) with scaling
        X_expanded[i, :] = by_data[i] * X_basis[i, :]
    """
    if by_type == 'factor':
        levels = pd.unique(by_data)
        k = len(levels)
        n, p = X_basis.shape
        
        X_expanded = np.zeros((n, k*p))
        for level_idx, level in enumerate(levels):
            mask = (by_data == level)
            X_expanded[mask, level_idx*p:(level_idx+1)*p] = X_basis[mask, :]
        
        return X_expanded, levels
    
    elif by_type == 'continuous':
        # Element-wise multiply
        return X_basis * by_data[:, np.newaxis], None
```

**Step 3: Penalty Expansion (~80 lines)**
```python
# In penalty_matrix.py

def expand_penalty_with_by(P_base, by_dim, by_type):
    """
    Expand penalty matrix for by-variable
    
    by_type='factor' with k levels:
        Output: block-diagonal matrix
        P_expanded = diag(P, P, ..., P)  [k times]
        Smoothing parameters can be shared or separate per level
    """
    if by_type == 'factor':
        # Block diagonal
        P_expanded = scipy.linalg.block_diag(*[P_base]*by_dim)
        return P_expanded
    
    elif by_type == 'continuous':
        # Scaled penalty (normalize by variation in by_data)
        # P_expanded = P_base (variance-weighted)
        return P_base
```

**Step 4: Coefficient Extraction & Interpretation (~70 lines)**
```python
# In api/gam.py (predict, summary methods)

def extract_by_effects(self, smooth_name):
    """
    Extract coefficients for each by-level separately
    Returns: dict {level_1: coef_1, level_2: coef_2, ...}
    """
    smooth = self.smooth_terms[smooth_name]
    if smooth.by_variable is not None:
        levels = smooth.by_levels
        coef_dim = smooth.basis_dimension
        
        by_coefs = {}
        for i, level in enumerate(levels):
            start = i * coef_dim
            end = (i + 1) * coef_dim
            by_coefs[level] = self.beta[start:end]
        
        return by_coefs
    else:
        return None

def predict_by_level(self, test_data, level=None):
    """
    Predictions for specific level (factor) or weighted by continuous by
    """
```

**Step 5: Visualization & Summary (~70 lines)**
```python
# In visualization/plot.py

def plot_by_effects(model, smooth_name, by_variable):
    """
    Overlay smooth effects for each level of by-variable
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots()
    by_effects = model.extract_by_effects(smooth_name)
    
    for level, coef in by_effects.items():
        # Predict on fine grid for this level
        pred = model.predict(..., level=level)
        axes.plot(pred['x'], pred['fit'], label=f'Level: {level}')
    
    axes.legend()
    return fig

# In api/summary.py
def summary_by_smooth(model, smooth_name):
    """
    Print separate significance tests per level
    
    Smooth term 's(x, by=group)':
      Level A: EDF=5.2, Ref.df=10, F=8.5, p=< 0.001 **
      Level B: EDF=4.8, Ref.df=10, F=6.2, p=< 0.001 **
      Level C: EDF=2.1, Ref.df=10, F=1.3, p=0.23 (ns)
    """
```

**Testing:**
```python
# tests/test_by_variables.py

def test_by_factor_basis_expansion():
    X = np.random.randn(100, 5)
    by = np.array(['A']*30 + ['B']*40 + ['C']*30)
    
    X_exp, levels = expand_basis_with_by(X, by, 'factor')
    assert X_exp.shape == (100, 15)  # 3 levels * 5 cols
    assert len(levels) == 3

def test_by_factor_fitting():
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'group': np.repeat(['A', 'B', 'C'], [40, 30, 30])
    })
    
    model = GAM('y ~ s(x, by=group)', data=df).fit()
    
    # Should have 3 separate smooth effects
    assert len(model.edf_by_level['s(x, by=group)']) == 3
    
    # Predictions differ per group
    df_A = df[df['group'] == 'A']
    df_B = df[df['group'] == 'B']
    pred_A = model.predict(df_A, level='A')
    pred_B = model.predict(df_B, level='B')
    assert not np.allclose(pred_A['fit'], pred_B['fit'])

def test_by_default_vs_R():
    # Ensure coefficients match R mgcv output
```

**Deliverables:**
- Extended `formula_parser.py` (by parsing)
- Extended `model_matrix.py` (axis expansion)
- Extended `penalty_matrix.py` (block-diagonal)
- Extended `api/summary.py` and `api/predict.py` (per-level extraction)
- Visualization updates
- Comprehensive tests

**Success Criteria:**
- ✅ Factor by-variables parse correctly
- ✅ Design matrix expands to correct shape (k*p)
- ✅ Fitting converges identically for each level
- ✅ Separate coefficients per level (can be extracted)
- ✅ Predictions differ correctly per group
- ✅ Coefficients match R mgcv to 1e-4

**Estimated Time:** 10–12 hours

**ROI:** +25–30 points on Specification (critical feature)

**Dependencies:** Task 1 (solver)

---

#### TASK 4: Implement Weights Support

**Priority:** 🔴 CRITICAL (needed for robust regression, survey data)

**Current State:**
- Files: `pymgcv/api/gam.py`, `pymgcv/optimizer/pirls.py`
- Status: Score 0/100 (not started)
- Impact: Case weights are common in applied work

**Files to Modify:**
1. **MODIFY:** `pymgcv/api/gam.py` (~30 lines to add weights parameter)
2. **MODIFY:** `pymgcv/optimizer/pirls.py` (~50 lines to weight likelihood)
3. **ADD TESTS:** `tests/test_weights.py` (~100 lines)

**Implementation Steps:**

**Step 1: GAM Class Enhancement (~30 lines)**
```python
# In api/gam.py

class GAM:
    def __init__(self, ..., weights=None):
        # NEW PARAMETER
        self.weights_column = weights
        self.weights = None  # Will be loaded in fit()
    
    def fit(self, data=None, ...):
        # Load weights from data
        if self.weights_column is not None:
            if isinstance(self.weights_column, str):
                self.weights = data[self.weights_column].values
            else:
                self.weights = self.weights_column
            
            # Validate
            assert np.all(self.weights > 0), "Weights must be positive"
            assert np.all(np.isfinite(self.weights)), "Weights must be finite"
            
            # Optional: normalize
            self.weights = self.weights / self.weights.mean()
        else:
            self.weights = np.ones(self.n)
```

**Step 2: PIRLS Integration (~50 lines)**
```python
# In optimizer/pirls.py

class PIRLSSolver:
    def solve(self, X, y, penalties, family, weights=None):
        """Solve with weights"""
        
        # Working weights: w = (dmu/deta)^2 / Var(mu)
        # If weights provided, multiply by external weights
        w = (dmu_deta**2) / var_mu  # [existing line]
        
        if weights is not None:
            # Element-wise multiply
            w = w * weights  # Shape (n,)
        
        # Create diagonal weight matrix
        W = np.diag(w)  # Or sparse for large n
        
        # Solve: (X'WX + Σ λ P) β = X'Wz
        # Instead of: (X'X + Σ λ P) β = X'z
        
        XtWX = X.T @ W @ X
        Xtwz = X.T @ (W @ z)
        
        # Add penalties (unchanged)
        XtWX += sum_penalties
        
        # Solve linear system (unchanged)
        beta = scipy.linalg.solve(XtWX, Xtwz, assume_a='pos')
        
        return beta
```

**Step 3: Alternative: Weighted Data Centering (~30 lines)**
```python
# More numerically stable approach:

def solve_with_weights(self, X, y, penalties, family, weights):
    """
    Alternative: Pre-multiply by sqrt(weights)
    
    Equivalent: solve (sqrt(W) X)'(sqrt(W) X) β = (sqrt(W) X)'(sqrt(W) z)
    """
    sqrt_w = np.sqrt(weights)
    
    X_weighted = X * sqrt_w[:, np.newaxis]
    z_weighted = z * sqrt_w
    
    # Standard PIRLS on weighted data
    XtX = X_weighted.T @ X_weighted
    Xtz = X_weighted.T @ z_weighted
    
    XtX += sum_penalties
    beta = scipy.linalg.solve(XtX, Xtz, assume_a='pos')
    
    return beta
```

**Step 4: EDF & Statistics Adjustment (~40 lines)**
```python
# In optimizer/edf.py

def compute_edf(self, X, penalties, weights=None):
    """
    Effective degrees of freedom with weights
    
    Hat matrix: H = X(X'WX + Σ λ P)^{-1}X'W
    EDF = trace(H)
    """
    # Solve for H matrix rows
    W = np.diag(weights) if weights is not None else np.eye(X.shape[0])
    
    # Only need diagonal of H for efficient trace
    edf_per_obs = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        # Solve for i-th row of (X'WX + λP)^{-1} X'W
        # edf_per_obs[i] = X[i] @ inv(X'WX + λP) @ X'[i] @ W[i,i]
        pass
    
    return edf_per_obs.sum()
```

**Step 5: Diagnostics & Interpretation (~50 lines)**
```python
# In diagnostics/residuals.py

def residuals_weighted(self, type='deviance'):
    """Residuals accounting for weights"""
    if self.weights is None:
        return self.residuals(type=type)
    
    res = self.y - self.fitted_values
    
    if type == 'standardized':
        # Weighted standardization
        se = self.se_fit / np.sqrt(self.weights)
        return res / se
    
    elif type == 'pearson':
        # Pearson with weight adjustment
        return res / (np.sqrt(self.dispersion) * np.sqrt(self.weights))

# In visualization/plot.py
def plot_residuals_with_weights(model):
    """
    Residual plot with bubble size ~ weight
    """
    fig, ax = plt.subplots()
    ax.scatter(model.fitted_values, model.residuals(),
               s=model.weights*50,  # Size by weight
               alpha=0.6)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Fitted')
    ax.set_ylabel('Residuals')
    return fig
```

**Testing:**
```python
# tests/test_weights.py

def test_weights_specification():
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'w': np.random.exponential(1, 100)
    })
    
    model = GAM('y ~ s(x)', data=df, weights='w')
    assert model.weights is not None
    assert len(model.weights) == 100

def test_unweighted_vs_uniform():
    # If all weights = 1, should get same result as unweighted
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'w': np.ones(100)
    })
    
    m1 = GAM('y ~ s(x)', data=df).fit()
    m2 = GAM('y ~ s(x)', data=df, weights='w').fit()
    
    assert np.allclose(m1.coef, m2.coef, atol=1e-6)

def test_weight_influence():
    # Down-weighted outliers should reduce their influence
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
    })
    df.loc[0, 'y'] = 100  # Big outlier
    
    m_equal = GAM('y ~ s(x)', data=df, weights=np.ones(100)).fit()
    weights_down = np.ones(100)
    weights_down[0] = 0.01  # Down-weight outlier
    m_robust = GAM('y ~ s(x)', data=df, weights=weights_down).fit()
    
    # Robust should fit better to rest of data
    residuals_robust = m_robust.residuals()
    residuals_equal = m_equal.residuals()
    # residuals_robust should have smaller variance
```

**Deliverables:**
- Modified `api/gam.py` (weights parameter)
- Modified `optimizer/pirls.py` (weight integration)
- Modified `optimizer/edf.py` (weighted EDF)
- Modified diagnostics and visualization (weight adjustment)
- Unit tests

**Success Criteria:**
- ✅ Weights parameter accepted and validated
- ✅ Unweighted data (uniform weights=1) gives identical result
- ✅ Down-weighted observations show reduced influence
- ✅ Coefficients match R mgcv to 1e-4
- ✅ EDF properly adjusted for weights

**Estimated Time:** 6–8 hours

**ROI:** +17–23 points on Specification

**Dependencies:** Task 1 (solver)

---

#### TASK 5: Complete Cubic Regression Spline Implementation

**Priority:** 🟠 HIGH (complete smooth bases set)

**Current State:**
- File: `pymgcv/smooth/cubic_spline.py`
- Status: Score 35/100 (partial skeleton)
- Impact: Fast, accurate univariate smooth (complement to TPRS)

**Files to Modify:**
1. **REPLACE:** `pymgcv/smooth/cubic_spline.py` (~250 lines, mostly rewrite)
2. **ADD TESTS:** `tests/test_cubic_spline.py` (~150 lines)

**Implementation Steps:**

**Step 1: Knot Placement (~80 lines)**
```python
# In cubic_spline.py

class CubicRegressionSpline:
    def __init__(self, x, num_knots=None, knots=None, boundary_order=3):
        """
        Cubic regression spline constructor
        
        Parameters:
        -----------
        x : array-like, shape (n,)
            Univariate predictor
        num_knots : int, optional
            Number of interior knots. Default: max(3, floor(sqrt(n)))
        knots : array-like, optional
            Manual knot specification
        boundary_order : int
            Order of boundary knots (typically 3 for natural cubic)
        """
        self.x = np.asarray(x)
        self.n = len(self.x)
        
        if knots is not None:
            self.knots = np.asarray(knots)
        elif num_knots is None:
            num_knots = max(3, int(np.sqrt(self.n)))
            self.knots = np.quantile(self.x, np.linspace(0, 1, num_knots+2)[1:-1])
        else:
            self.knots = np.quantile(self.x, np.linspace(0, 1, num_knots+2)[1:-1])
        
        # Full knot vector with boundaries
        min_x = self.x.min()
        max_x = self.x.max()
        
        self.tknots = np.concatenate([
            [min_x]*boundary_order,
            self.knots,
            [max_x]*boundary_order
        ])
        
        self.basis_dim = len(self.knots) + 2
```

**Step 2: Cubic Basis Construction (~100 lines)**
```python
# Use spline.BasisSpline or manually

from scipy.interpolate import CubicSpline

def basis_matrix(self, x_eval):
    """
    Construct cubic regression spline basis matrix
    
    Returns:
    --------
    X : array, shape (len(x_eval), basis_dim)
        Basis matrix
    """
    x_eval = np.asarray(x_eval)
    n_eval = len(x_eval)
    X = np.zeros((n_eval, self.basis_dim))
    
    # Basis function j is cubic spline with knots, non-zero mainly on [t_j, t_{j+3}]
    # Use recursion or explicit formulation
    
    for j in range(self.basis_dim):
        # Create j-th basis function
        # Option 1: Use scipy's B-spline basis
        # Option 2: Use finite element approach (cubic polynomials per interval)
        
        # Simple approach: cubic Hermite on each knot interval
        for i, x_val in enumerate(x_eval):
            X[i, j] = self._basis_j(x_val, j)
    
    return X

def _basis_j(self, x, j):
    """Evaluate j-th basis function at x"""
    # Natural cubic spline approach:
    # Each interval uses cubic poly with C2 continuity at knots
    # Total 4*num_intervals parameters
    # Constrained by: continuity + second derivative continuity
    # Results in (num_knots + 2) basis functions
    pass
```

**Step 3: Penalty Matrix (~120 lines)**
```python
def penalty_matrix(self, order=2):
    """
    Construct penalty matrix (integrated squared derivatives)
    
    For order=2: ∫ f''(x)^2 dx
    
    Computed via numerical integration (Gaussian quadrature)
    over knot intervals
    """
    P = np.zeros((self.basis_dim, self.basis_dim))
    
    # Gauss quadrature: integrate over each knot interval
    from scipy.integrate import quad
    
    for i in range(self.basis_dim):
        for j in range(self.basis_dim):
            # Integrate product of j-th and j-th derivatives
            
            def integrand(x):
                bi = self._basis_derivative(x, i, order)
                bj = self._basis_derivative(x, j, order)
                return bi * bj
            
            # Integrate over range where both are non-zero
            P[i, j], _ = quad(integrand, self.x.min(), self.x.max())
    
    return P

def _basis_derivative(self, x, j, order=1):
    """Evaluate order-th derivative of j-th basis at x"""
    # Finite difference approximation or analytical (complicated)
    h = 1e-6
    if order == 1:
        return (self._basis_j(x+h, j) - self._basis_j(x-h, j)) / (2*h)
    elif order == 2:
        return (self._basis_j(x+h, j) - 2*self._basis_j(x, j) + self._basis_j(x-h, j)) / h**2
```

**Step 4: Comparison with TPRS (~50 lines)**
```python
def validate_vs_tprs(self, y, x_smooth, x_tprs):
    """
    Compare cubic spline with TPRS for same data
    
    Should be similar with many knots in cubic
    """
    model_cubic = GAM('y ~ s(x, bs="cr")').fit()
    model_tprs = GAM('y ~ s(x, bs="tp")').fit()
    
    # Compare deviances, EDF, smoothing parameters
    assert abs(model_cubic.dev - model_tprs.dev) / model_tprs.dev < 0.1
    assert abs(model_cubic.edf[0] - model_tprs.edf[0]) < 1.0
```

**Testing:**
```python
def test_cubic_spline_knots():
    x = np.linspace(0, 1, 100)
    cs = CubicRegressionSpline(x, num_knots=10)
    assert len(cs.knots) == 10
    assert cs.basis_dim == 12  # knots + 2

def test_cubic_basis_shape():
    x = np.linspace(0, 1, 100)
    cs = CubicRegressionSpline(x, num_knots=10)
    X = cs.basis_matrix(x)
    assert X.shape == (100, 12)

def test_cubic_fit_quality():
    # Fit smooth sine curve
    x = np.linspace(0, 1, 100)
    y = np.sin(2*np.pi*x) + np.random.randn(100)*0.1
    
    model = GAM('y ~ s(x, bs="cr")').fit()
    y_pred = model.predict(x)
    
    # Should fit well
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
    assert r2 > 0.9
```

**Deliverables:**
- Full `pymgcv/smooth/cubic_spline.py` implementation
- Unit tests
- Examples comparing with TPRS

**Success Criteria:**
- ✅ Basis matrix correct shape and partition of unity
- ✅ Penalty matrix is SPD
- ✅ Fitting converges for typical data
- ✅ Coefficients match R mgcv to 1e-4
- ✅ EDF calculation correct

**Estimated Time:** 10–12 hours

**ROI:** +15–20 points on Smooth Bases

**Dependencies:** Task 1 (solver)

---

#### TASK 6: Implement Fixed Smoothing Parameters (sp=)

**Priority:** 🟠 HIGH (enabler for reproducibility, grid search)

**Current State:**
- File: `pymgcv/api/gam.py`
- Status: Score partially addressed in formula parsing (65/100)
- Impact: Simple to implement, unlocks reproducible workflows

**Implementation Steps:**

**Step 1: Parameter Addition (~20 lines)**
```python
# In api/gam.py

class GAM:
    def __init__(self, ..., sp=None, ...):
        """
        Parameters:
        -----------
        sp : array-like, optional
            Fixed smoothing parameters. If provided, MAGIC optimization is skipped.
            Length must equal number of smooth terms.
        """
        self.sp_fixed = sp
        self.sp_optimization_skipped = False
    
    def fit(self, ...):
        if self.sp_fixed is not None:
            # Validate
            num_smooths = len(self.smooth_terms)
            if len(self.sp_fixed) != num_smooths:
                raise ValueError(f"sp has {len(self.sp_fixed)} values, "
                                f"but model has {num_smooths} smooth terms")
            
            # Skip MAGIC optimizer
            self.smoothing_parameters = self.sp_fixed
            self.sp_optimization_skipped = True
        else:
            # Standard MAGIC optimization
            self.smoothing_parameters = self.magic_optimizer.optimize(...)
            self.sp_optimization_skipped = False
```

**Step 2: MAGIC Bypass (~30 lines)**
```python
# In api/gam.py fit() method

def fit(self, ...):
    # ... setup ...
    
    if self.sp_fixed is not None:
        # Single PIRLS iteration with fixed λ
        logger.info(f"Using fixed smoothing parameters: {self.sp_fixed}")
        
        # Assemble penalties with fixed λ
        P_penalized = sum(lambda_j * P_j for lambda_j, P_j in 
                         zip(self.sp_fixed, self.penalty_matrices))
        
        # Solve: (X'WX + P_penalized) β = X'Wz
        self.beta = self.pirls_solver.solve(X, y, self.density_matrices,
                                            self.family,
                                            max_iter=1,  # Single iteration only
                                            custom_penalty=P_penalized)
        
        self.fitted = True
        
    else:
        # Standard path: MAGIC optimization
        self.magic_optimizer.optimize(...)
        self.fitted = True
```

**Step 3: Documentation (~30 lines)**
```python
# In docstring

Example:
    >>> # Reproducible fitting with fixed λ
    >>> model = GAM('y ~ s(x1) + s(x2)').fit()
    >>> sp_values = model.smoothing_parameters
    
    >>> # Refit with same λ (reproducible)
    >>> model2 = GAM('y ~ s(x1) + s(x2)', sp=sp_values).fit()
    >>> np.allclose(model.coef, model2.coef)
    True
    
    >>> # Manual λ selection
    >>> model3 = GAM('y ~ s(x1) + s(x2)', sp=[0.1, 1.0]).fit()
    
    >>> # Grid search
    >>> results = []
    >>> for lambda_1 in [0.01, 0.1, 1.0, 10.0]:
    ...     for lambda_2 in [0.01, 0.1, 1.0]:
    ...         m = GAM(formula, sp=[lambda_1, lambda_2]).fit()
    ...         results.append({'sp': [lambda_1, lambda_2], 'aic': m.aic})
```

**Testing:**
```python
def test_fixed_sp():
    df = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'y': np.random.randn(100)
    })
    
    model_auto = GAM('y ~ s(x1) + s(x2)', data=df).fit()
    sp_auto = model_auto.smoothing_parameters
    
    # Fit again with same sp
    model_fixed = GAM('y ~ s(x1) + s(x2)', sp=sp_auto, data=df).fit()
    
    # Should get same coefficients
    assert np.allclose(model_fixed.coef, model_auto.coef, atol=1e-6)

def test_fixed_sp_validation():
    formula = 'y ~ s(x1) + s(x2)'
    sp_wrong = [0.1]  # Only 1 value, but formula has 2 smooths
    
    with pytest.raises(ValueError):
        GAM(formula, sp=sp_wrong)
```

**Deliverables:**
- Modified `api/gam.py` with sp parameter
- Updated docstrings with examples
- Unit tests

**Success Criteria:**
- ✅ sp parameter accepted
- ✅ MAGIC optimization skipped when sp provided
- ✅ Fixed sp produces identical results on re-fit
- ✅ Works with all family types

**Estimated Time:** 3–4 hours

**ROI:** +8–13 points on Specification

**Dependencies:** None (independent)

---

## END OF TASK-BY-TASK BREAKDOWN

See IMPLEMENTATION_ROADMAP_TO_98_PARITY.md for:
- Tasks 7–16 (B-Splines, P-Splines, gam.check, Confidence Intervals, etc.)
- Overall sequencing and phase planning
- Effort allocation across teams
- Risk management and testing strategy

