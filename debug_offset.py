import numpy as np
from pymgcv.optimizer.pirls import PIRLSSolver
from pymgcv.distributions.family_base import GaussianFamily

np.random.seed(42)
n = 40
X = np.column_stack([np.ones(n), np.random.randn(n)])

true_beta = np.array([1.0, 0.5])
offset = np.full(n, 0.5)
y = X @ true_beta + offset + np.random.randn(n) * 0.1

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"offset shape: {offset.shape}")
print(f"true_beta: {true_beta}")
print(f"y[:5] = {y[:5]}")
print(f"X @ true_beta[:5] + offset[:5] = {(X @ true_beta + offset)[:5]}")

family = GaussianFamily()
S_list = [np.zeros((2, 2))]

solver = PIRLSSolver(X, y, family, S_list, offset=offset)

# Print initial state
print(f"\nInitial beta: {solver.beta}")
print(f"Initial eta: {solver.eta[:5]}")
print(f"Initial mu: {solver.mu[:5]}")

# Manually run first iteration to debug
beta_init = solver.beta.copy()

# Manual iteration
solver.eta = solver.X @ solver.beta + solver.offset
solver.mu = solver.family.linkinv(solver.eta)
print(f"\nAfter eta/mu computation:")
print(f"eta[:5]: {solver.eta[:5]}")
print(f"mu[:5]: {solver.mu[:5]}")

dmu_deta = solver.family.dmu_deta(solver.eta)
var_mu = solver.family.variance(solver.mu, solver.dispersion)
print(f"dmu_deta[:5]: {dmu_deta[:5]}")
print(f"var_mu[:5]: {var_mu[:5]}")

var_mu = np.maximum(var_mu, 1e-10)
dmu_deta = np.where(np.abs(dmu_deta) < 1e-10, 1e-10, dmu_deta)

w = solver.weights * (dmu_deta**2) / var_mu
z = solver.eta + solver.weights * (solver.y - solver.mu) / dmu_deta

print(f"\nWeights: {w[:5]}")
print(f"Working vector z[:5]: {z[:5]}")

XtWX = solver.X.T @ (solver.X * w[:, np.newaxis])
Xtwz = solver.X.T @ (w * z)
S = solver._construct_combined_penalty()

print(f"\nX^T W X:\n{XtWX}")
print(f"X^T W z: {Xtwz}")
print(f"S:\n{S}")

A = XtWX + S
print(f"A = X^T W X + S:\n{A}")

from scipy import linalg
beta_new = linalg.solve(A, Xtwz)
print(f"\nSolved beta_new: {beta_new}")
print(f"Difference from true: {beta_new - true_beta}")

# Now run the full solver
print("\n" + "="*50)
print("Full solver run:")
beta = solver.solve(max_iter=20, verbose=True)

print(f"\nRecovered beta: {beta}")
print(f"True beta: {true_beta}")
print(f"Difference: {beta - true_beta}")
print(f"Solver converged: {solver.converged}")

# Check if within tolerance
print(f"\nAllclose with atol=0.3: {np.allclose(beta, true_beta, atol=0.3)}")
print(f"Allclose with atol=0.5: {np.allclose(beta, true_beta, atol=0.5)}")

