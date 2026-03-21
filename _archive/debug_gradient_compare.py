"""Compare gradient at same lambda using warm-started vs fresh PIRLS."""
import numpy as np
import pandas as pd
from pymgcv.optimizer.reml_objective import REMLObjective
from pymgcv.optimizer.pirls import PIRLSSolver
from pymgcv.distributions.family_base import PoissonFamily
from pymgcv.utils.model_matrix import ModelMatrix
from pymgcv.penalties.penalty_matrix import PenaltyMatrix

RNG = np.random.default_rng(42)
x = np.linspace(0, 3, 200)
y = RNG.poisson(np.exp(0.5 + 0.6 * x)).astype(float)
df = pd.DataFrame({'x': x, 'y': y})

mm = ModelMatrix(df, 'y ~ s(x)')
X = mm.X
y_arr = df['y'].values.astype(float)
p = X.shape[1]
family = PoissonFamily()
off = 1
S_list = []
for bobj in mm.smooth_bases:
    k = bobj.basis.shape[1] if hasattr(bobj, 'basis') else 10
    if hasattr(bobj, 'penalty_matrix'):
        S_raw = bobj.penalty_matrix()
    else:
        S_raw = PenaltyMatrix(basis_dim=k, penalty_type='tprs').S
    S_embed = np.zeros((p, p))
    k2 = min(S_raw.shape[0], p - off)
    S_embed[off:off+k2, off:off+k2] = S_raw[:k2, :k2]
    S_list.append(S_embed)
    off += k

reml_obj = REMLObjective(X, y_arr, family, S_list, [1], [p - 1])

# Solve PIRLS at lambda=0.087 (initial)
lam1 = 0.087
beta1 = PIRLSSolver(X, y_arr, family, S_list, lambda_vec=np.array([lam1]),
                    offset=np.zeros(len(y_arr))).solve(max_iter=200, tol=1e-12)

# Solve PIRLS at lambda=1.74 from cold start
lam2 = 1.74
beta2_cold = PIRLSSolver(X, y_arr, family, S_list, lambda_vec=np.array([lam2]),
                         offset=np.zeros(len(y_arr))).solve(max_iter=200, tol=1e-12)

# Solve PIRLS at lambda=1.74 warm-started from beta1
beta2_warm = PIRLSSolver(X, y_arr, family, S_list, lambda_vec=np.array([lam2]),
                         offset=np.zeros(len(y_arr))).solve(max_iter=200, tol=1e-12,
                                                              beta_init=beta1)

ll2 = np.array([np.log(lam2)])

score_cold, grad_cold, H_cold = reml_obj.objective_gradient_hessian(beta2_cold, ll2)
score_warm, grad_warm, H_warm = reml_obj.objective_gradient_hessian(beta2_warm, ll2)

print(f"At lambda=1.74:")
print(f"  Cold-start PIRLS: REML={score_cold:.4f}, grad={float(grad_cold[0]):.4f}")
print(f"  Warm-start PIRLS: REML={score_warm:.4f}, grad={float(grad_warm[0]):.4f}")
print(f"  Beta diff (cold vs warm): max |Δβ| = {np.max(np.abs(beta2_cold - beta2_warm)):.2e}")
print(f"  beta2_cold[:5]: {beta2_cold[:5]}")
print(f"  beta2_warm[:5]: {beta2_warm[:5]}")
print()

# Also check: are these betas actually PIRLS-converged?
# PIRLS stationarity: X'W(z - Xb) = S_lambda * b  where z = eta + (y-mu)/dmu_deta
for name, beta_t, lam_t in [('cold', beta2_cold, lam2), ('warm', beta2_warm, lam2)]:
    eta = X @ beta_t
    mu = np.exp(eta)
    dmu_deta = mu  # Poisson
    var_mu = mu
    w = (dmu_deta**2) / var_mu
    z = eta + (y_arr - mu) / dmu_deta
    
    XtWX = X.T @ (X * w[:, None])
    Xtwz = X.T @ (w * z)
    S_lam = lam_t * S_list[0]
    A = XtWX + S_lam
    rhs = Xtwz
    lhs = A @ beta_t
    resid = np.max(np.abs(lhs - rhs))
    print(f"  PIRLS convergence check ({name}): ||Aβ - X'Wz|| = {resid:.2e}")

# What does the gradient FORMULA give for these two betas?
print("\nDetailed gradient breakdown at lambda=1.74:")
for name, beta_t in [('cold', beta2_cold), ('warm', beta2_warm)]:
    from scipy import linalg
    from pymgcv.linalg.penalized_solver import PenalizedSolver
    
    lam_vec = np.array([lam2])
    S_comb = lam2 * S_list[0]
    
    eta = X @ beta_t
    mu = np.exp(eta)
    w = mu  # Poisson: dmu/deta = mu, V(mu) = mu, so w = mu^2/mu = mu
    
    XtWX = X.T @ (X * w[:, None])
    psolver = PenalizedSolver(XtWX, S_comb)
    AinvS = psolver.solve(S_list[0])
    
    trace_term = lam2 * float(np.trace(AinvS))
    penalty_term = lam2 * float(beta_t @ S_list[0] @ beta_t)
    rank = len([e for e in np.linalg.eigvalsh(S_list[0]) if e > 1e-10])
    
    grad_manual = penalty_term + trace_term - rank
    print(f"  {name}: trace_term={trace_term:.4f}, penalty_term={penalty_term:.4f}, rank={rank}, grad={grad_manual:.4f}")
