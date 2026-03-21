import pandas as pd
import numpy as np
from pymgcv.api.gam import GAM

df = pd.read_csv('insurance_loss_cost_data.csv')

formula = (
    'capped_claims_kusd ~ offset(log_duration) '
    '+ s(vehicle_age,k=8) + s(driver_age,k=10) '
    '+ s(bonus_malus,k=8) + s(log10_exposure_usd,k=6) '
    '+ s(annual_mileage_km,k=6) '
    '+ class_B + class_C + class_D + is_urban + is_suburban'
)

m = GAM(formula, data=df, family='tweedie', method='reml', gamma=1.2)
m.fit(df, verbose=False)

print(f'Total EDF  : {m.edf:.4f}')
print(f'Dispersion : {m.dispersion_:.4f}')
print(f'Lambda     : {np.round(m.smoothing_parameters, 2)}')
print()
print('Per-smooth EDF:')
for k, v in m.edf_per_smooth.items():
    print(f'  {k}: {v["edf"]:.4f}')

y = df['capped_claims_kusd'].values
mu_fit = m.predict(df, scale='response')
mu_null = np.full(len(y), y.mean())
# deviance = -2 * (loglik(y, mu) - loglik(y, y)) approx
dev_null = -2.0 * m.family.loglik(y, mu_null, m.dispersion_)
dev_fit  = -2.0 * m.family.loglik(y, mu_fit, m.dispersion_)
dev_explained = (1.0 - dev_fit / dev_null) * 100
print()
print(f'Deviance explained: {dev_explained:.2f}%')
print()
print('R mgcv targets:')
print('  Total EDF (smooth sum) : 5.6076')
print('  Deviance explained     : 3.4%')
print('  Dispersion             : ~12')
