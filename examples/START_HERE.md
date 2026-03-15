# START HERE: How to Use PyMGCV in Your Own Programs

## Quick Answer

1. **Read:** [HOW_TO_USE_PYMGCV.md](../HOW_TO_USE_PYMGCV.md) (5 min read)
2. **Copy:** [COPY_AND_MODIFY_THIS.py](COPY_AND_MODIFY_THIS.py) to your project folder
3. **Modify:** Change the 3 sections marked "MODIFY THIS"
4. **Run:** `python your_file.py`

Done! 🎉

---

## File Guide

### For First-Time Users

| File | Purpose | Read Time |
|------|---------|-----------|
| **HOW_TO_USE_PYMGCV.md** | Complete guide to using pymgcv | 10 min |
| **COPY_AND_MODIFY_THIS.py** | Simplest template to copy | 5 min |
| **MY_FIRST_GAM.py** | Well-commented example | 10 min |

### For Learning by Examples

| File | Purpose | Complexity |
|------|---------|------------|
| **MY_FIRST_GAM.py** | Basic Gaussian GAM | Beginner |
| **QUICK_START_GUIDE.py** | 7 detailed examples | Intermediate |
| **simple_gam_demo.py** | Output formatting | Intermediate |
| **comparison_with_R.py** | Advanced comparison | Advanced |

---

## Three Ways to Get Started

### Way 1: The Fastest (5 minutes)

```bash
# 1. Copy the template
cp examples/COPY_AND_MODIFY_THIS.py my_analysis.py

# 2. Edit MY_ANALYSIS.PY
#    - Change Section 1: load_your_data()
#    - Change Section 2: create_model()
#    - Keep Section 3 as is!

# 3. Run it
python my_analysis.py
```

### Way 2: Learn by Example (20 minutes)

```bash
# 1. Read HOW_TO_USE_PYMGCV.md
cat HOW_TO_USE_PYMGCV.md

# 2. Run MY_FIRST_GAM.py to see it in action
python examples/MY_FIRST_GAM.py

# 3. Run QUICK_START_GUIDE.py for 7 examples
python examples/QUICK_START_GUIDE.py

# 4. Copy MY_FIRST_GAM.py and customize it
cp examples/MY_FIRST_GAM.py my_analysis.py
# Then edit it for your data
```

### Way 3: Full Deep Dive (1 hour)

```bash
# 1. Read all documentation
cat HOW_TO_USE_PYMGCV.md
cat COMPARISON_GUIDE.md

# 2. Run all examples
python examples/MY_FIRST_GAM.py
python examples/QUICK_START_GUIDE.py
python examples/simple_gam_demo.py

# 3. Study one example
cat examples/MY_FIRST_GAM.py

# 4. Build your own
# Copy MY_FIRST_GAM.py and customize
```

---

## Your New Python Program - Step by Step

### Step 1: Create Your File

```bash
# Create a new Python file
echo. > my_gam_analysis.py
```

Or in your editor (VS Code, PyCharm, etc.):
- File → New File → name it `my_gam_analysis.py`

### Step 2: Copy Starter Code

Copy this into `my_gam_analysis.py`:

```python
#!/usr/bin/env python
"""My GAM Analysis"""

import sys
sys.path.insert(0, 'c:/Users/surya/Downloads/pymgcv')

import numpy as np
import pandas as pd
from pymgcv.api.gam import GAM

# MODIFY: Load your data
data = pd.read_csv('your_data.csv')  # Change this!

# MODIFY: Create your model
model = GAM('y ~ s(x)', family='gaussian')  # Change this!

# Run (no modification needed)
model.fit(data)
print(model.summary())
print("\nPredictions:")
new_data = pd.DataFrame({'x': [0, 0.5, 1.0]})
print(model.predict(new_data))
```

### Step 3: Modify for Your Needs

**Change 1: Data Loading**
```python
# Option A: CSV
data = pd.read_csv('C:/path/to/your/file.csv')

# Option B: Excel
data = pd.read_excel('C:/path/to/your/file.xlsx')

# Option C: Synthetic
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y = np.sin(6*np.pi*x) + np.random.normal(0, 0.1, n)
data = pd.DataFrame({'x': x, 'y': y})
```

**Change 2: Model Formula**
```python
# Simple smooth
model = GAM('y ~ s(x)')

# Multiple smooths
model = GAM('y ~ s(x1) + s(x2)')

# With parametric term
model = GAM('y ~ z + s(x)')

# Poisson regression
model = GAM('y ~ s(x)', family='poisson')

# Custom basis dimension
model = GAM('y ~ s(x, k=15)', family='gaussian')
```

**Change 3: Family**
```python
family='gaussian'   # Continuous (default)
family='poisson'    # Count data
family='binomial'   # Binary data
family='gamma'      # Positive continuous
family='tweedie'    # Insurance claims
```

### Step 4: Run Your Program

```bash
python my_gam_analysis.py
```

---

## Copy-Paste Examples

### Example 1: Simple Linear Relationship

```python
import sys
sys.path.insert(0, 'c:/Users/surya/Downloads/pymgcv')

import numpy as np
import pandas as pd
from pymgcv.api.gam import GAM

# Data: simple linear
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2*x + 5 + np.random.normal(0, 2, 100)
data = pd.DataFrame({'x': x, 'y': y})

# Fit
model = GAM('y ~ s(x, k=5)', family='gaussian')
model.fit(data)
print(model.summary())
```

### Example 2: Non-linear Relationship

```python
# Data: sine wave
np.random.seed(42)
x = np.linspace(0, 2*np.pi, 150)
y = np.sin(x) + np.random.normal(0, 0.2, 150)
data = pd.DataFrame({'x': x, 'y': y})

# Fit with smooth spline
model = GAM('y ~ s(x, k=15)', family='gaussian')
model.fit(data)
print(model.summary())

# Predict
x_new = np.linspace(0, 2*np.pi, 50)
new_data = pd.DataFrame({'x': x_new})
predictions = model.predict(new_data)
```

### Example 3: Multiple Variables

```python
# Data
np.random.seed(42)
n = 200
x1 = np.linspace(0, 1, n)
x2 = np.linspace(0, 2, n)
y = np.sin(6*np.pi*x1) + np.cos(4*np.pi*x2) + np.random.normal(0, 0.2, n)
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# Fit GAM with two smooth terms
model = GAM('y ~ s(x1) + s(x2)', family='gaussian')
model.fit(data)
print(model.summary())
```

### Example 4: Count Data (Poisson)

```python
# Count data
np.random.seed(42)
n = 150
x = np.linspace(0, 5, n)
lambda_param = np.exp(0.5 + 0.3*x)
y = np.random.poisson(lambda_param)
data = pd.DataFrame({'x': x, 'y': y})

# Poisson GAM
model = GAM('y ~ s(x, k=12)', family='poisson')
model.fit(data)
print(model.summary())
```

### Example 5: Mixed Model

```python
# Data with parametric and smooth components
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
z = np.random.normal(0, 1, n)
y = 1.5 + 2*z + np.sin(6*np.pi*x) + np.random.normal(0, 0.2, n)
data = pd.DataFrame({'x': x, 'z': z, 'y': y})

# Model: linear z + smooth x
model = GAM('y ~ z + s(x)', family='gaussian')
model.fit(data)
print(model.summary())

# The coefficient 2*z should be close to 2
if hasattr(model, 'coefficients'):
    print(f"\nEstimated coefficient for z: {model.coefficients[1]:.4f}")
```

---

## Common Tasks

### Task 1: Load your CSV file

```python
import pandas as pd
data = pd.read_csv('C:/Users/surya/Documents/mydata.csv')
print(data.head())  # Check it loaded correctly
```

### Task 2: Fit a model

```python
from pymgcv.api.gam import GAM

model = GAM('y ~ s(x)', family='gaussian')
model.fit(data)
```

### Task 3: View results

```python
# Full summary
print(model.summary())

# Individual components
print("Coefficients:", model.coefficients)
print("EDF:", model.edf)
print("AIC:", model.aic)
```

### Task 4: Make predictions

```python
new_data = pd.DataFrame({'x': [0.1, 0.5, 0.9]})
predictions = model.predict(new_data)
print(predictions)
```

### Task 5: Save results

```python
# Save predictions to CSV
results = pd.DataFrame({
    'x': new_data['x'],
    'predictions': predictions
})
results.to_csv('predictions.csv', index=False)

# Save model summary to file
with open('model_summary.txt', 'w') as f:
    f.write(model.summary())
```

---

## Troubleshooting

### "No module named 'pymgcv'"

Add this at the top:
```python
import sys
sys.path.insert(0, 'c:/Users/surya/Downloads/pymgcv')
```

Or install properly:
```bash
cd c:\Users\surya\Downloads\pymgcv
pip install -e .
```

### "KeyError: 'y'" or "KeyError: 'x'"

Check your column names:
```python
print(data.columns)  # See what columns you have
print(data.head())   # See the data
```

Make sure your formula matches your columns:
```python
# If your data has columns: 'response', 'predictor1'
# Use: model = GAM('response ~ s(predictor1)')
```

### Model not converging

Try:
- Scaling your data: `data = (data - data.mean()) / data.std()`
- Using fewer basis functions: `s(x, k=5)` instead of `s(x, k=20)`
- Using a simpler family: try `'gaussian'` first

---

## File Comparison

| Task | Use This File |
|------|---------------|
| Start immediately | COPY_AND_MODIFY_THIS.py |
| Learn by examples | MY_FIRST_GAM.py |
| Full reference | HOW_TO_USE_PYMGCV.md |
| 7 detailed examples | QUICK_START_GUIDE.py |
| Advanced usage | comparison_with_R.py |

---

## Next: Run Your First Program

1. Open `examples/COPY_AND_MODIFY_THIS.py`
2. Modify the 3 marked sections
3. Save to: `my_first_analysis.py`
4. Run: `python my_first_analysis.py`

That's it! You're ready to use pymgcv! 🎉

---

## Get More Help

- **Basic usage:** See HOW_TO_USE_PYMGCV.md
- **More examples:** Run QUICK_START_GUIDE.py
- **Validation:** See COMPARISON_GUIDE.md
- **Questions:** Check docstrings: `help(GAM)` or `help(model.fit)`
