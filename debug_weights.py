import numpy as np
from pymgcv.optimizer.pirls import PIRLSSolver
from pymgcv.distributions.family_base import GaussianFamily

np.random.seed(42)
n = 60
X = np.column_stack([np.ones(n), np.random.randn(n)])

# Heteroscedastic data
y = X @ np.array([1.0, 0.5])
sigma = 0.1 + 0.5 * X[:, 1]**2
y += np.random.randn(n) * sigma

# Weights should be inverse variance
weights = 1.0 / sigma

family = GaussianFamily()
S_list = [np.zeros((2, 2))]

# Weighted fit
solver_wgt = PIRLSSolver(X, y, family, S_list, weights=weights)
beta_wgt = solver_wgt.solve(max_iter=20, verbose=False)

print(f"Weighted solver converged: {solver_wgt.converged}")
print(f"Weighted beta: {beta_wgt}")
print(f"Number of iterations: {len(solver_wgt.dev_history)}")
print(f"Dev history (first 5): {solver_wgt.dev_history[:5] if len(solver_wgt.dev_history) > 5 else solver_wgt.dev_history}")
print(f"Dev history (last 5): {solver_wgt.dev_history[-5:] if len(solver_wgt.dev_history) > 5 else solver_wgt.dev_history}")

