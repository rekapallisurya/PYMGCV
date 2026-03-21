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
print(f'Lambda     : {m.smoothing_parameters}')
print()
print('Per-smooth EDF:')
for k, v in m.edf_per_smooth.items():
    edf_val = v['edf']
    print(f'  {k}: {edf_val:.4f}')

y = df['capped_claims_kusd'].values
mu_null = np.full(len(y), y.mean())
dev_null = m.family.deviance(y, mu_null, 1.0)
dev_fit  = m.family.deviance(y, m.fitted_values_, 1.0)
print()
print(f'Deviance explained: {(1 - dev_fit/dev_null)*100:.2f}%')
print(f'AIC        : {m.aic_:.2f}')
print()
print('Per-smooth EDF:')
for k, v in m.edf_per_smooth.items():
    edf_val = v['edf']
    print(f'  {k}: {edf_val:.4f}')

y = df['capped_claims_kusd'].values
mu_null = np.full(len(y), y.mean())
dev_null = m.family.deviance(y, mu_null, 1.0)
dev_fit  = m.family.deviance(y, m.fitted_values_, 1.0)
print()
print(f'Deviance explained: {(1 - dev_fit/dev_null)*100:.2f}%')
