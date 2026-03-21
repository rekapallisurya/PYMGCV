"""Deep trace: print lambda, gradient, Newton step at each outer iteration."""
import numpy as np
import pandas as pd
from pymgcv.optimizer.magic_optimizer import MAGICOptimizer
from pymgcv.optimizer.reml_objective import REMLObjective
from pymgcv.optimizer.pirls import PIRLSSolver
from pymgcv.distributions.family_base import PoissonFamily, GaussianFamily
from pymgcv.utils.model_matrix import ModelMatrix
from pymgcv.penalties.penalty_matrix import PenaltyMatrix
from scipy import linalg

def build_parts(formula, df, family_obj):
    mm = ModelMatrix(df, formula)
    X = mm.X
    y = df['y'].values.astype(float)
    p = X.shape[1]
    S_list = []
    off = 1
    for bobj in mm.smooth_bases:
        k = bobj.basis.shape[1] if hasattr(bobj, 'basis') else 10
        if hasattr(bobj, 'penalty_matrix'):
            S_raw = bobj.penalty_matrix()
        elif hasattr(bobj, 'penalty_matrix_S'):
            S_raw = bobj.penalty_matrix_S()
        else:
            S_raw = PenaltyMatrix(basis_dim=k, penalty_type='tprs').S
        S_embed = np.zeros((p, p))
        k2 = min(S_raw.shape[0], p - off)
        S_embed[off:off+k2, off:off+k2] = S_raw[:k2, :k2]
        S_list.append(S_embed)
        off += k
    return X, y, S_list


RNG = np.random.default_rng(42)
x = np.linspace(0, 3, 200)
y = RNG.poisson(np.exp(0.5 + 0.6 * x)).astype(float)
df = pd.DataFrame({'x': x, 'y': y})

X, y_arr, S_list = build_parts('y ~ s(x)', df, PoissonFamily())
family = PoissonFamily()
n = len(y_arr)

# Reproduce the MAGICOptimizer initial lambda
y_var = max(float(np.var(y_arr)), 1e-10)
lam_init_list = []
for S_j in S_list:
    eig_max = np.max(np.abs(np.linalg.eigvalsh(S_j)))
    if eig_max > 1e-10:
        lam_init = y_var / eig_max
        lam_init_list.append(max(lam_init, 1e-10))
print(f"y_var = {y_var:.4f}")
print(f"Initial lambda: {lam_init_list}")

# Manual Newton loop with detailed prints
lambda_log = np.array([np.log(lam_init_list[0])])
reml_obj = REMLObjective(X, y_arr, family, S_list, [1], [X.shape[1]-1])

print(f"\n{'iter':>4}  {'lambda':>12}  {'REML':>12}  {'grad':>10}  {'H':>10}  {'Newton-step':>12}  {'alpha':>6}  {'action'}")
print('-' * 90)

prev_beta = None
for outer_it in range(50):
    lam = np.exp(lambda_log[0])
    solver = PIRLSSolver(X, y_arr, family, S_list, lambda_vec=np.exp(lambda_log),
                         offset=np.zeros(n))
    beta = solver.solve(max_iter=50, tol=1e-9, beta_init=prev_beta)
    prev_beta = beta.copy()
    
    score, grad, hess = reml_obj.objective_gradient_hessian(beta, lambda_log)
    
    # Newton step
    try:
        d = linalg.solve(hess, -grad)
    except Exception:
        d = -0.01 * grad
    
    # Clip
    d_orig = d.copy()
    max_step = 3.0
    step_scale = np.max(np.abs(d))
    if step_scale > max_step:
        d = d * max_step / step_scale
    
    # Backtracking
    alpha = 1.0
    accepted = False
    for bt in range(10):
        ll_new = np.clip(lambda_log + alpha * d, -20, 20)
        lam_new = np.exp(ll_new)
        s_new = PIRLSSolver(X, y_arr, family, S_list, lambda_vec=lam_new,
                            offset=np.zeros(n))
        beta_new = s_new.solve(max_iter=50, tol=1e-9, beta_init=prev_beta)
        score_new = reml_obj.objective(beta_new, ll_new)
        if score_new < score:
            lambda_log = ll_new
            accepted = True
            break
        alpha *= 0.5
    
    if not accepted:
        # fallback: take smallest alpha step anyway
        lambda_log = np.clip(lambda_log + alpha * d, -20, 20)
    
    action = 'ACCEPT' if accepted else 'FALLBACK'
    step_taken = float(np.abs(alpha * d[0]))
    print(f"{outer_it:>4}  {lam:>12.4f}  {score:>12.3f}  {float(grad[0]):>10.4f}  {float(hess[0,0]):>10.4f}  {float(d_orig[0]):>12.4f}  {alpha:>6.4f}  {action}  step={step_taken:.2e}")
    
    if step_taken < 1e-5:
        print("CONVERGED (step < 1e-5)")
        break

print(f"\nFinal lambda = {float(np.exp(lambda_log[0])):.4f}")
print(f"mgcv optimal = 43989")
