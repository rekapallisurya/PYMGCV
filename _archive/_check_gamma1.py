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

# Test with gamma=1.0 to check if that gives better convergence
m = GAM(formula, data=df, family='tweedie', method='reml', gamma=1.0)
m.fit(df, verbose=False)

print(f'--- gamma=1.0 ---')
print(f'Total EDF  : {m.edf:.4f}')
print(f'Dispersion : {m.dispersion_:.4f}')
print(f'Lambda     : {np.round(m.smoothing_parameters, 4)}')
print()
print('Per-smooth EDF:')
for k, v in m.edf_per_smooth.items():
    print(f'  {k}: {v["edf"]:.4f}')

mu_fit = m.predict(df, scale='response')
y = df['capped_claims_kusd'].values
p = m.family.power

# Proper Tweedie unit deviance (no phi, no Wright)
dev_arr = np.zeros(len(y))
nz = y > 0
dev_arr[nz] = 2 * (y[nz] * (y[nz]**(1-p) - mu_fit[nz]**(1-p)) / (1-p) + (mu_fit[nz]**(2-p) - y[nz]**(2-p)) / (2-p))
dev_arr[~nz] = 2 * mu_fit[~nz]**(2-p) / (2-p)

mu_null = np.full(len(y), y.mean())
dev_null_arr = np.zeros(len(y))
dev_null_arr[nz] = 2 * (y[nz] * (y[nz]**(1-p) - mu_null[nz]**(1-p)) / (1-p) + (mu_null[nz]**(2-p) - y[nz]**(2-p)) / (2-p))
dev_null_arr[~nz] = 2 * mu_null[~nz]**(2-p) / (2-p)

dev_fit_total = dev_arr.sum()
dev_null_total = dev_null_arr.sum()
print(f'Deviance explained: {(1 - dev_fit_total/dev_null_total)*100:.2f}%')

print()
print('R mgcv targets (gamma=1.2):')
print('  Total EDF (smooth sum) : 5.6076')
print('  Deviance explained     : 3.4%')
print('  Dispersion             : ~12')
