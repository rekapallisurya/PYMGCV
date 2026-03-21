"""Profile REML objective over a range of lambda to understand the landscape."""
import numpy as np
import pandas as pd
from pymgcv.optimizer.magic_optimizer import MAGICOptimizer
from pymgcv.optimizer.reml_objective import REMLObjective
from pymgcv.optimizer.pirls import PIRLSSolver
from pymgcv.distributions.family_base import PoissonFamily, GaussianFamily
from pymgcv.utils.model_matrix import ModelMatrix
from pymgcv.utils.formula_parser import FormulaParser
from pymgcv.penalties.penalty_matrix import PenaltyMatrix

def get_model_parts(formula, df, family_str):
    """Get X, y, S_list etc from a fitted GAM layout."""
    from pymgcv.api.gam import GAM
    g = GAM(formula, data=df, family=family_str, method='REML')
    # Build the model matrix only (don't fit)
    g.data = df
    parser = FormulaParser(formula)
    mm = ModelMatrix(df, formula)
    X = mm.matrix
    y = df['y'].values.astype(float)
    
    # Build S_list  
    S_list = []
    p = X.shape[1]
    off = 1  # skip intercept
    for i, spec in enumerate(parser.smooth_terms):
        bobj = mm.smooth_bases[i]
        k = bobj.basis.shape[1] if hasattr(bobj, 'basis') else 10
        s_stop = off + k
        
        if hasattr(bobj, 'penalty_matrix'):
            S_raw = bobj.penalty_matrix()
        elif hasattr(bobj, 'penalty_matrix_S'):
            S_raw = bobj.penalty_matrix_S()
        else:
            from pymgcv.penalties.penalty_matrix import PenaltyMatrix
            S_raw = PenaltyMatrix(basis_dim=k, penalty_type='tprs').S
        
        S_embed = np.zeros((p, p))
        k2 = min(k, p - off)
        S_embed[off:off+k2, off:off+k2] = S_raw[:k2, :k2]
        S_list.append(S_embed)
        off = s_stop
    
    return X, y, S_list


RNG = np.random.default_rng(42)

# Poisson data
x_p = np.linspace(0, 3, 200)
y_p = RNG.poisson(np.exp(0.5 + 0.6 * x_p)).astype(float)
df_p = pd.DataFrame({'x': x_p, 'y': y_p})

# CR data
RNG2 = np.random.default_rng(42)
x_c = np.linspace(0, 1, 200)
y_c = np.sin(2 * np.pi * x_c) + 0.4 * RNG2.normal(0, 1, 200)
df_c = pd.DataFrame({'x': x_c, 'y': y_c})


def profile_reml(df, formula, family_obj, family_str, lambda_range):
    """Evaluate REML at each lambda in lambda_range."""
    from pymgcv.utils.model_matrix import ModelMatrix
    mm = ModelMatrix(df, formula)
    X = mm.X
    y = df['y'].values.astype(float)
    p = X.shape[1]

    # Get penalty matrix
    S_list = []
    off = 1
    for i, bobj in enumerate(mm.smooth_bases):
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

    reml_obj = REMLObjective(X, y, family_obj, S_list, [1], [p-1])

    print(f"\n  lambda       REML       grad       sp_step")
    print(f"  --------     -------    -------    -------")
    for lam in lambda_range:
        log_lam = np.array([np.log(lam)])
        # PIRLS at this lambda
        pirls = PIRLSSolver(X, y, family_obj, S_list, lambda_vec=np.array([lam]),
                            offset=np.zeros_like(y))
        beta = pirls.solve(max_iter=50)
        score, grad, hess = reml_obj.objective_gradient_hessian(beta, log_lam)
        # Newton step
        try:
            step = float(np.linalg.solve(hess, -grad)[0])
        except Exception:
            step = float('nan')
        print(f"  {lam:10.3f}  {score:10.3f}  {grad[0]:10.4f}  {step:10.4f}")


print("=== POISSON REML PROFILE ===")
print("(mgcv optimal: lambda ~= 43989)")
profile_reml(df_p, 'y ~ s(x)', PoissonFamily(), 'poisson',
             [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 44000])

print("\n=== CR (GAUSSIAN) REML PROFILE ===")
print("(mgcv optimal: lambda ~= 11.64)")
profile_reml(df_c, "y ~ s(x, bs='cr')", GaussianFamily(), 'gaussian',
             [0.001, 0.01, 0.1, 0.5, 1, 5, 11, 12, 50, 100])
