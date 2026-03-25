# Quickstart

This guide walks you through fitting your first GAM with pymgcv.

## 1. Basic GAM

```python
import numpy as np
import pandas as pd
from pymgcv import GAM

# Simulate data
np.random.seed(42)
n = 200
x = np.linspace(0, 2 * np.pi, n)
y = np.sin(x) + np.random.normal(0, 0.3, n)
df = pd.DataFrame({"x": x, "y": y})

# Fit a GAM with a smooth term
model = GAM("y ~ s(x)", data=df)
model.fit()

# Print R-style summary
print(model.summary())

# Predictions
preds = model.predict(df, scale="response")
```

## 2. Multiple Smooth Terms

```python
df = pd.DataFrame({
    "y": y,
    "x1": np.random.randn(n),
    "x2": np.random.randn(n),
    "x3": np.random.randn(n),
})

model = GAM("y ~ s(x1) + s(x2, k=15) + s(x3, bs='cr')", data=df)
model.fit()
print(model.summary())
```

## 3. Tweedie GLM (Insurance Pricing)

```python
from pymgcv import GAM, Tweedie

model = GAM(
    "loss_cost ~ s(driver_age) + s(vehicle_age) + s(exposure)",
    family=Tweedie(p=1.5),
    data=insurance_df,
)
model.fit()
print(model.summary())
```

## 4. Diagnostics

```python
from pymgcv import gam_check

gam_check(model)  # Residual QQ plot, k-index test, etc.
```

## 5. Visualization

```python
from pymgcv import plot_gam

plot_gam(model)  # Plot all smooth effects
```

## Next Steps

- [GAM Modeling Guide](../user-guide/gam-modeling.md) — smooth types, families, tensor products
- [Comparison with R mgcv](../user-guide/comparison-with-r.md) — parity report
- [API Reference](../api/index.md) — full API docs
