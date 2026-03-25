"""Scan income_k lambda to check REML surface."""
import pandas as pd
import numpy as np
from pymgcv import GAM, s, Tweedie
from pymgcv.distributions.family_base import TweedieFamily
from pymgcv.optimizer.pirls import PIRLSSolver
from pymgcv.optimizer.reml_objective import REMLObjective
from pymgcv.optimizer.edf import EDFComputer

df = pd.read_csv('campaign_data.csv')
df['income_k'] = df['income'] / 1000

model = GAM('spend ~ s(age) + s(income_k) + s(contacts)', data=df,
            family=Tweedie(link='log', power=1.331))
model.fit()

X = model._X_fit; y = model._y_fit
offset = np.zeros(len(y))

S_list = []; smooth_starts = []; smooth_sizes = []
for j in range(len(model.model_matrix.smooth_bases)):
    sl = model.model_matrix.smooth_indices[j]
    basis = model.model_matrix.smooth_bases[j]
    P = np.zeros((X.shape[1], X.shape[1]))
    P[sl.start:sl.stop, sl.start:sl.stop] = basis.S
    S_list.append(P)
    smooth_starts.append(sl.start)
    smooth_sizes.append(sl.stop - sl.start)

fam = TweedieFamily(power=1.331)
base_lam = model.smoothing_parameters[:3].copy()
print(f'Baseline lambda: {base_lam}')
print(f'Scanning income_k lambda (smooth 1)...')
header = f"{'lam_inc':>12}  {'REML':>12}  {'EDF_tot':>8}  {'bSb_inc':>10}"
print(header)

for mult in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]:
    lam_test = base_lam.copy()
    lam_test[1] *= mult

    solver = PIRLSSolver(X, y, fam, S_list, lambda_vec=lam_test,
                         offset=offset, dispersion=1.0)
    beta = solver.solve(max_iter=50, tol=1e-6, verbose=False)

    reml_obj = REMLObjective(X, y, fam, S_list,
                             smooth_starts=smooth_starts,
                             smooth_sizes=smooth_sizes,
                             offset=offset, dispersion=1.0)
    reml_score = reml_obj.objective(beta, np.log(lam_test))

    S_comb = sum(l * S for l, S in zip(lam_test, S_list))
    edf_comp = EDFComputer(X, S_comb, fam, beta, offset, dispersion=1.0)
    edf = edf_comp.total_edf()

    bSb = float(beta @ S_list[1] @ beta)
    print(f'{lam_test[1]:12.4f}  {reml_score:12.2f}  {edf:8.2f}  {bSb:10.4f}')
