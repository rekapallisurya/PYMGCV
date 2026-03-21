"""Check that gamma=1.2 (user-specified) runs stably with the gamma cap at 1.0."""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

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

print('--- gamma=1.2 (optimizer internally caps to 1.0) ---')
print(f'Total EDF  : {m.edf:.4f}')
print(f'Dispersion : {m.dispersion_:.4f}')
print(f'Lambda     : {m.smoothing_parameters.round(4)}')

smooth_edf = sum(v['edf'] for v in m.edf_per_smooth.values())
print(f'Smooth EDF sum: {smooth_edf:.4f}  (R mgcv target: 5.6076)')

mu = m.predict(df, scale='response')
y = df['capped_claims_kusd'].values
p = m.family.power

d = np.zeros(len(y))
nz = y > 0
d[nz] = 2 * (y[nz] * (y[nz]**(1-p) - mu[nz]**(1-p)) / (1-p)
             + (mu[nz]**(2-p) - y[nz]**(2-p)) / (2-p))
d[~nz] = 2 * mu[~nz]**(2-p) / (2-p)

mu0 = np.full(len(y), y.mean())
d0 = np.zeros(len(y))
d0[nz] = 2 * (y[nz] * (y[nz]**(1-p) - mu0[nz]**(1-p)) / (1-p)
              + (mu0[nz]**(2-p) - y[nz]**(2-p)) / (2-p))
d0[~nz] = 2 * mu0[~nz]**(2-p) / (2-p)

print(f'Deviance explained: {(1 - d.sum()/d0.sum())*100:.2f}%  (R mgcv target: 3.4%)')

# Clip check: lambda should NOT be at clip max (5e8)
clip_max = 5e8
hit_clip = np.sum(m.smoothing_parameters >= clip_max * 0.99)
print(f'\nLambda at clip max: {hit_clip}/5  (should be 0)')
