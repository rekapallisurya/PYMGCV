"""Side-by-side comparison script: pymgcv vs mgcv for each test case."""
import numpy as np
import pandas as pd
import subprocess, json, tempfile, os

R_EXE = r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe"
RNG = np.random.default_rng(42)

# Same data generators as validate_pymgcv.py
def make_data():
    rng = np.random.default_rng(42)
    datasets = {}
    n = 200
    
    x = np.linspace(0, 1, n)
    y = 2 + 3*x + 2*np.sin(4*np.pi*x) + rng.normal(0, 0.3, n)
    datasets['gaussian'] = ('y ~ s(x)', 'gaussian', pd.DataFrame({'x': x, 'y': y}))
    
    x = np.linspace(0, 3, n)
    eta = 0.5 + 0.6*x
    y = rng.poisson(np.exp(eta))
    datasets['poisson'] = ('y ~ s(x)', 'poisson', pd.DataFrame({'x': x, 'y': y.astype(float)}))
    
    x = np.linspace(0, 1, n)
    p = 1/(1+np.exp(-(2*np.sin(4*np.pi*x))))
    y = rng.binomial(1, p).astype(float)
    datasets['binomial'] = ('y ~ s(x)', 'binomial', pd.DataFrame({'x': x, 'y': y}))
    
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 2, n)
    y = 1.2*np.sin(4*np.pi*x1) + 0.8*np.cos(3*np.pi*x2) + rng.normal(0, 0.2, n)
    datasets['multi'] = ('y ~ s(x1) + s(x2)', 'gaussian', pd.DataFrame({'x1': x1, 'x2': x2, 'y': y}))
    
    x = np.linspace(0, 1, n)
    y = np.sin(2*np.pi*x) + 0.4*rng.normal(0, 1, n)
    datasets['cr'] = ("y ~ s(x, bs='cr')", 'gaussian', pd.DataFrame({'x': x, 'y': y}))
    
    return datasets

datasets = make_data()

# R script template
R_TEMPLATE = """
suppressMessages(library(mgcv))
suppressMessages(library(jsonlite))

{data_writes}

results <- list()
{fits}
writeLines(toJSON(results, auto_unbox=TRUE, digits=15), con="{out}")
"""

# Write all data and fits
data_writes = []
fits = []
csv_paths = {}
for name, (formula, family, df) in datasets.items():
    tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
    df.to_csv(tmp.name, index=False)
    tmp.close()
    csv_paths[name] = tmp.name
    R_family = '' if family == 'gaussian' else f', family={family}()'
    data_writes.append(f'd_{name} <- read.csv("{tmp.name.replace(chr(92), "/")}")')
    fits.append(f'''
m_{name} <- gam({formula}, data=d_{name}, method="REML"{R_family})
results${name} <- list(
  sp=as.numeric(m_{name}$sp), edf=as.numeric(sum(m_{name}$edf)),
  deviance=as.numeric(m_{name}$deviance),
  fitted=as.numeric(fitted(m_{name}))[1:5]
)''')

out_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
r_script = R_TEMPLATE.format(
    data_writes='\n'.join(data_writes),
    fits='\n'.join(fits),
    out=out_file.replace('\\', '/')
)
r_tmp = tempfile.NamedTemporaryFile(suffix='.R', delete=False, mode='w')
r_tmp.write(r_script)
r_tmp.close()

# Run R
result = subprocess.run([R_EXE, r_tmp.name], capture_output=True, text=True)
if result.returncode != 0:
    print("R ERRORS:", result.stderr[:2000])
else:
    r_results = json.loads(open(out_file).read())
    print("mgcv reference results:")
    for name, res in r_results.items():
        print(f"  {name}: sp={res['sp']}, edf={res['edf']:.3f}, "
              f"dev={res['deviance']:.4f}, fitted[:3]={[round(f,4) for f in res['fitted'][:3]]}")

# Now run pymgcv
print("\npymgcv results:")
from pymgcv.api.gam import GAM

for name, (formula, family, df) in datasets.items():
    try:
        m = GAM(formula, data=df, family=family, method='REML')
        m.fit()
        eta = m._X_fit @ m.beta
        if m.model_matrix.offset_vector() is not None:
            eta += m.model_matrix.offset_vector()
        mu = m.family.linkinv(eta)
        sp = m.smoothing_parameters
        edf = m.edf
        dev = -2.0 * m.family.loglik(m._y_fit, mu, m.dispersion_)
        fitted_first3 = [round(float(mu[i]), 4) for i in range(3)]
        print(f"  {name}: sp={[round(float(s), 4) for s in sp]}, edf={edf:.3f}, "
              f"dev={dev:.4f}, fitted[:3]={fitted_first3}")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

print("\nCleaning up...")
for path in csv_paths.values():
    try: os.unlink(path)
    except: pass
try: os.unlink(out_file) ; os.unlink(r_tmp.name)
except: pass
