"""Verbose debug: REML optimizer trace for Poisson and CR."""
import numpy as np
import pandas as pd
from pymgcv.api.gam import GAM

RNG = np.random.default_rng(42)

# --- Poisson ---
x = np.linspace(0, 3, 200)
y = RNG.poisson(np.exp(0.5 + 0.6 * x)).astype(float)
df = pd.DataFrame({'x': x, 'y': y})

print("=== POISSON VERBOSE (50 outer iters) ===")
m = GAM('y ~ s(x)', data=df, family='poisson', method='REML',
        control={'maxit': 50, 'trace': True})
m.fit()
print(f"Final sp={m.smoothing_parameters}, edf={m.edf:.4f}\n")

# --- CR (cubic regression spline) ---
RNG2 = np.random.default_rng(42)
x = np.linspace(0, 1, 200)
y2 = np.sin(2 * np.pi * x) + 0.4 * RNG2.normal(0, 1, 200)
df2 = pd.DataFrame({'x': x, 'y': y2})

print("=== CR VERBOSE (50 outer iters) ===")
m2 = GAM("y ~ s(x, bs='cr')", data=df2, family='gaussian', method='REML',
         control={'maxit': 50, 'trace': True})
m2.fit()
print(f"Final sp={m2.smoothing_parameters}, edf={m2.edf:.4f}")
print("mgcv target sp=11.64, edf=7.84")
