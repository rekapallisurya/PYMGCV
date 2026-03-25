"""Profile REML across Tweedie power values to understand landscape."""
import pandas as pd
import numpy as np
from pymgcv import GAM, s, Tweedie
from pymgcv.distributions.family_base import TweedieFamily
from pymgcv.optimizer.magic_optimizer import MAGICOptimizer
from pymgcv.optimizer.reml_objective import REMLObjective

df = pd.read_csv('campaign_data.csv')
df['income_k'] = df['income'] / 1000

print(f"{'p':>6}  {'REML':>12}  {'deviance':>12}  {'loglik':>12}  {'edf':>8}")
for p_val in [1.01, 1.1, 1.2, 1.344, 1.5, 1.7, 1.9]:
    model = GAM('spend ~ s(age) + s(income_k) + s(contacts)', data=df,
                family=Tweedie(link='log', power=p_val))
    model.fit()
    
    X = model._X_fit; y = model._y_fit
    offset = model.model_matrix.offset_vector()
    if offset is None:
        offset = np.zeros(len(y))
    eta = X @ model.beta + offset
    mu = model.family.linkinv(eta)
    dev = model.family.deviance(y, mu)
    ll = model.family.loglik(y, mu, 1.0)
    
    # REML is stored in the optimizer but not exposed. Compute it manually
    # Actually let's get the REML from a fresh computation
    S_list = []
    smooth_starts = []
    smooth_sizes = []
    for j_sm in range(len(model.model_matrix.smooth_bases)):
        sm_slice = model.model_matrix.smooth_indices[j_sm]
        basis_obj = model.model_matrix.smooth_bases[j_sm]
        s_start = sm_slice.start; s_stop = sm_slice.stop
        p_total = X.shape[1]
        P_embed = np.zeros((p_total, p_total))
        P_embed[s_start:s_stop, s_start:s_stop] = basis_obj.S
        S_list.append(P_embed)
        smooth_starts.append(s_start)
        smooth_sizes.append(s_stop - s_start)
    
    reml_obj = REMLObjective(X, y, model.family, S_list, 
                             smooth_starts=smooth_starts, smooth_sizes=smooth_sizes,
                             offset=offset, dispersion=1.0)
    reml_score = reml_obj.objective(model.beta, np.log(model.smoothing_parameters[:len(S_list)]))
    
    print(f'{p_val:6.3f}  {reml_score:12.1f}  {dev:12.1f}  {ll:12.1f}  {model.edf:8.2f}')
