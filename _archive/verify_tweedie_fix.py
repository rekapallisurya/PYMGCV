import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pymgcv.api.gam import GAM

df = pd.read_csv('insurance_loss_cost_data.csv')
y  = df['capped_claims_kusd'].values
d  = df['duration'].values
p  = 1.5

# Analytic MLE for reference
b_ana = np.sum(y / d**(p-1)) / np.sum(d**(2-p))
print(f'Analytic null MLE: exp(b)={b_ana:.4f}  pred_mean={b_ana*d.mean():.4f}')

# Null model
m0 = GAM('capped_claims_kusd ~ offset(log_duration)',
         data=df, family='tweedie', sp=[], control={'maxit': 30})
m0.fit()
pred0 = m0.predict(df, scale='response')
print(f'Null model: beta0={float(m0.beta[0]):.4f}  exp(b0)={np.exp(float(m0.beta[0])):.4f}  '
      f'pred_mean={pred0.mean():.4f}  bal={pred0.sum()/y.sum():.4f}')

# 2-smooth model
m1 = GAM(
    'capped_claims_kusd ~ offset(log_duration) + s(driver_age,k=8) + s(bonus_malus,k=8)',
    data=df, family='tweedie', method='REML', gamma=1.0, control={'maxit': 100})
m1.fit()
pred1 = m1.predict(df, scale='response')
print(f'2-smooth model: EDF={m1.edf:.2f}  bal={pred1.sum()/y.sum():.4f}  '
      f'pred_mean={pred1.mean():.4f}')
for t, v in m1.edf_per_smooth.items():
    print(f'  {t}: edf={v["edf"]:.2f}')
