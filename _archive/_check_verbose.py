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
m.fit(df, verbose=True)

print()
print(f'Total EDF  : {m.edf:.4f}')
print(f'Dispersion : {m.dispersion_:.4f}')
print(f'Lambda (log): {np.log(m.smoothing_parameters)}')
print(f'Lambda     : {m.smoothing_parameters}')
