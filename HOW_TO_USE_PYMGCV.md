# How to Use PyMGCV in Your Own Python Programs

## Method 1: Simple Path Setup (Easiest for Development)

### Step 1: Create Your New Python File

```python
# my_analysis.py

import sys
import numpy as np
import pandas as pd

# Add pymgcv to Python path
sys.path.insert(0, 'c:/Users/surya/Downloads/pymgcv')

# Now import pymgcv
from pymgcv.api.gam import GAM

# ... rest of your code ...
```

### Step 2: Run Your Program

```bash
python my_analysis.py
```

---

## Method 2: Install as Editable Package (Recommended)

### Step 1: Install pymgcv

```bash
# Navigate to pymgcv directory
cd c:\Users\surya\Downloads\pymgcv

# Install in editable mode
pip install -e .
```

### Step 2: Import Without Path Setup

```python
# my_analysis.py

import numpy as np
import pandas as pd
from pymgcv.api.gam import GAM  # Direct import!

# ... rest of your code ...
```

### Step 3: Run from Anywhere

```bash
python my_analysis.py
```

---

## Basic Program Structure

Here's the recommended structure for any pymgcv analysis:

```python
#!/usr/bin/env python
"""
My GAM Analysis
Description of what this program does.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

# Add pymgcv to path (if not installed via pip)
sys.path.insert(0, 'c:/Users/surya/Downloads/pymgcv')

# Import pymgcv
from pymgcv.api.gam import GAM

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

def load_data() -> pd.DataFrame:
    """Load or generate your data."""
    
    # Option A: Generate synthetic data
    np.random.seed(42)
    n = 150
    x = np.linspace(0, 1, n)
    y = np.sin(6*np.pi*x) + np.random.normal(0, 0.1, n)
    
    # Option B: Load from CSV
    # data = pd.read_csv('mydata.csv')
    
    # Option C: Load from Excel
    # data = pd.read_excel('mydata.xlsx')
    
    data = pd.DataFrame({'x': x, 'y': y})
    return data


# ============================================================================
# PART 2: MODEL FITTING
# ============================================================================

def fit_gam_model(data: pd.DataFrame) -> GAM:
    """Fit a GAM model to the data."""
    
    # Create model
    model = GAM(
        formula='y ~ s(x, k=10)',
        family='gaussian'
    )
    
    # Fit model
    model.fit(data, verbose=False)
    
    return model


# ============================================================================
# PART 3: RESULTS ANALYSIS
# ============================================================================

def analyze_results(model: GAM) -> None:
    """Print and analyze model results."""
    
    print("\n" + "="*70)
    print("MODEL RESULTS")
    print("="*70)
    
    # Print summary
    if hasattr(model, 'summary'):
        print(model.summary())
    
    # Extract key quantities
    print("\nKey Quantities:")
    if hasattr(model, 'coefficients'):
        print(f"  Intercept: {model.coefficients[0]:.6f}")
    if hasattr(model, 'edf'):
        print(f"  EDF: {model.edf}")
    if hasattr(model, 'aic'):
        print(f"  AIC: {model.aic:.2f}")


# ============================================================================
# PART 4: PREDICTIONS
# ============================================================================

def make_predictions(model: GAM) -> dict:
    """Make predictions on new data."""
    
    # Create prediction data
    x_new = np.linspace(0, 1, 50)
    pred_data = pd.DataFrame({'x': x_new})
    
    # Get predictions
    if hasattr(model, 'predict'):
        predictions = model.predict(pred_data)
        return {
            'x': x_new,
            'predictions': predictions
        }
    
    return None


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Main analysis pipeline."""
    
    print("\n" + "#"*70)
    print("My GAM Analysis")
    print("#"*70)
    
    # Load data
    print("\n[1] Loading data...")
    data = load_data()
    print(f"    Loaded {len(data)} observations")
    
    # Fit model
    print("\n[2] Fitting GAM model...")
    model = fit_gam_model(data)
    print("    ✓ Model fitted")
    
    # Analyze results
    print("\n[3] Analyzing results...")
    analyze_results(model)
    
    # Make predictions
    print("\n[4] Making predictions...")
    pred_results = make_predictions(model)
    if pred_results:
        print(f"    ✓ Generated {len(pred_results['x'])} predictions")
    
    print("\n" + "#"*70)
    print("Analysis complete!")
    print("#"*70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    main()
```

---

## Common Formulas & Syntax

### Smooth Terms

```python
# Single smooth
GAM('y ~ s(x)')

# Smooth with basis dimension
GAM('y ~ s(x, k=10)')

# Multiple smooths
GAM('y ~ s(x1) + s(x2)')

# Tensor product (2D smooth)
GAM('y ~ te(x1, x2)')

# Parametric + smooth
GAM('y ~ z + s(x)')  # z is linear, s(x) is smooth
```

### Families

```python
# Gaussian (default)
GAM('y ~ s(x)', family='gaussian')

# Poisson (count data)
GAM('y ~ s(x)', family='poisson')

# Binomial (binary data)
GAM('y ~ s(x)', family='binomial')

# Gamma (positive continuous)
GAM('y ~ s(x)', family='gamma')

# Tweedie (insurance claims)
GAM('y ~ s(x)', family='tweedie')
```

---

## Working with Your Data

### Option 1: CSV File

```python
import pandas as pd
from pymgcv.api.gam import GAM

# Load data
data = pd.read_csv('mydata.csv')

# Check first few rows
print(data.head())

# Fit model
model = GAM('y ~ s(x)')
model.fit(data)
```

### Option 2: Excel File

```python
import pandas as pd

# Load data
data = pd.read_excel('mydata.xlsx', sheet_name='Sheet1')

# Fit model
model = GAM('y ~ s(x)')
model.fit(data)
```

### Option 3: Generate Synthetic Data

```python
import numpy as np
import pandas as pd

# Generate data
np.random.seed(42)
n = 200
x = np.linspace(0, 1, n)
y = np.sin(6*np.pi*x) + np.random.normal(0, 0.1, n)

# Create DataFrame
data = pd.DataFrame({'x': x, 'y': y})

# Fit model
model = GAM('y ~ s(x)')
model.fit(data)
```

### Option 4: Extract Columns from Existing Data

```python
# Load full dataset
full_data = pd.read_csv('large_dataset.csv')

# Extract columns of interest
data = full_data[['response_col', 'predictor_col1', 'predictor_col2']].copy()
data.columns = ['y', 'x1', 'x2']

# Fit model
model = GAM('y ~ s(x1) + s(x2)')
model.fit(data)
```

---

## Accessing Model Results

```python
# Coefficients
model.coefficients        # Array of coefficients

# Standard errors
model.se                  # Standard error of each coefficient

# Effective degrees of freedom
model.edf                 # Dict of EDF per smooth term

# Smoothing parameters
model.lambda_             # Dict of λ per smooth term

# Deviance
model.deviance            # Deviance (goodness of fit)

# AIC
model.aic                 # Akaike Information Criterion

# GCV
model.gcv                 # Generalized Cross-Validation

# Formula
model.formula             # The formula string

# Family
model.family              # The family name

# Summary
model.summary()           # Full formatted output

# Predictions
model.predict(new_data)   # Predictions on new data
```

---

## Complete Working Example

Save this as `my_first_gam.py`:

```python
#!/usr/bin/env python
"""My first PyMGCV GAM analysis"""

import sys
sys.path.insert(0, 'c:/Users/surya/Downloads/pymgcv')

import numpy as np
import pandas as pd
from pymgcv.api.gam import GAM

# Generate data
np.random.seed(42)
n = 150
x = np.linspace(0, 2*np.pi, n)
y = np.sin(x) + 0.1*x + np.random.normal(0, 0.3, n)

# Create DataFrame
data = pd.DataFrame({'x': x, 'y': y})
print(f"Data shape: {data.shape}")

# Fit model
print("\nFitting GAM...")
model = GAM('y ~ s(x, k=10)', family='gaussian')
model.fit(data, verbose=False)
print("✓ Model fitted!")

# Print results
if hasattr(model, 'summary'):
    print("\n" + model.summary())

# Make predictions
x_new = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
new_data = pd.DataFrame({'x': x_new})

if hasattr(model, 'predict'):
    predictions = model.predict(new_data)
    print("\nPredictions:")
    for xi, pred in zip(x_new, predictions):
        print(f"  x={xi:6.4f}: {pred:10.6f}")
```

Run it:
```bash
python my_first_gam.py
```

---

## Troubleshooting

### Error: "No module named 'pymgcv'"

**Solution:** Add path or install:
```python
import sys
sys.path.insert(0, 'c:/Users/surya/Downloads/pymgcv')
```

Or install properly:
```bash
cd c:\Users\surya\Downloads\pymgcv
pip install -e .
```

### Error: "No module named 'pymgcv.api.gam'"

**Solution:** Check that pymgcv is properly installed:
```bash
pip list | grep pymgcv
```

### Error: "ValueError: DataFrame has NaN values"

**Solution:** Clean your data:
```python
# Remove NaN
data = data.dropna()

# Or fill NaN
data = data.fillna(data.mean())
```

### Model Not Fitting

**Solution:** Check data format:
```python
print(data.head())
print(data.dtypes)
print(data.describe())
```

---

## File Organization

Recommended folder structure:

```
my_project/
├── my_analysis.py           # Your main analysis
├── my_data.csv              # Your data
├── results.csv              # Output results
└── README.md                # Documentation
```

---

## Next Steps

1. **Copy the template** (`MY_FIRST_GAM.py` or `QUICK_START_GUIDE.py`)
2. **Modify for your data** (load your CSV/Excel)
3. **Adjust the formula** (change predictors and basis dimensions)
4. **Run and inspect results**
5. **Make predictions** on new data sets

---

## For More Examples

See:
- `examples/QUICK_START_GUIDE.py` - 7 detailed examples
- `examples/MY_FIRST_GAM.py` - Minimal starter template
- `examples/simple_gam_demo.py` - Basic usage demo
- `examples/comparison_with_R.py` - Advanced usage

---

## Getting Help

If you need help:
1. Check the examples in `examples/` directory
2. Read docstrings: `help(GAM)` or `help(model.fit)`
3. Review the COMPARISON_GUIDE.md for more context
4. Check the TEST files for usage patterns

Happy modeling! 🎉
