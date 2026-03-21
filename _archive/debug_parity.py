"""Check fitted values at different lambdas for Poisson and check TPRS basis scaling."""
import numpy as np
import pandas as pd
from pymgcv.api.gam import GAM
from pymgcv.distributions.family_base import PoissonFamily
from pymgcv.utils.model_matrix import ModelMatrix
from pymgcv.optimizer.pirls import PIRLSSolver
from pymgcv.penalties.penalty_matrix import PenaltyMatrix

RNG = np.random.default_rng(42)
x = np.linspace(0, 3, 200)
y = RNG.poisson(np.exp(0.5 + 0.6 * x)).astype(float)
df = pd.DataFrame({'x': x, 'y': y})

print("=== Fitted values comparison: pymgcv fitted at various sp ===")
for sp in [0.85, 10, 40, 100, 1000, 44000]:
    m = GAM('y ~ s(x)', data=df, family='poisson', method='REML', sp=[sp])
    m.fit()
    mu = m.family.linkinv(m._X_fit @ m.beta)
    print(f"  sp={sp:>8}: fitted[0]={mu[0]:.4f}, fitted[99]={mu[99]:.4f}, fitted[199]={mu[199]:.4f}, edf={m.edf:.3f}")

print("\nmgcv reference:")
print("  sp=43989: fitted[0]=1.603, fitted[99]=4.033, edf=2.001")
print()

# Check the penalty matrix structure
print("=== TPRS penalty matrix eigenvalues (Poisson model) ===")
mm = ModelMatrix(df, 'y ~ s(x)')
X = mm.X
p = X.shape[1]
for i, bobj in enumerate(mm.smooth_bases):
    k = bobj.basis.shape[1] if hasattr(bobj, 'basis') else 10
    if hasattr(bobj, 'penalty_matrix'):
        S_raw = bobj.penalty_matrix()
    else:
        S_raw = PenaltyMatrix(basis_dim=k, penalty_type='tprs').S
    eigs = sorted(np.linalg.eigvalsh(S_raw), reverse=True)
    print(f"  basis {i}: shape={S_raw.shape}, max_eig={eigs[0]:.4f}, min_nonzero={min(e for e in eigs if e > 1e-10):.4f}, rank={sum(e > 1e-10 for e in eigs)}")
    print(f"  all eigs: {[f'{e:.3f}' for e in eigs[:10]]}")

R_EXE = r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe"
RNG_SEED = 42


def make_datasets():
    rng = np.random.default_rng(RNG_SEED)
    datasets = {}

    # Poisson
    x = np.linspace(0, 3, 200)
    y = rng.poisson(np.exp(0.5 + 0.6 * x)).astype(float)
    datasets['poisson'] = pd.DataFrame({'x': x, 'y': y})

    # Binomial
    rng2 = np.random.default_rng(RNG_SEED)
    x = np.linspace(0, 1, 200)
    p = 1 / (1 + np.exp(-(2 * np.sin(4 * np.pi * x))))
    y = rng2.binomial(1, p).astype(float)
    datasets['binomial'] = pd.DataFrame({'x': x, 'y': y})

    # Gaussian
    rng3 = np.random.default_rng(RNG_SEED)
    x = np.linspace(0, 1, 200)
    y = 2 + 3 * x + 2 * np.sin(4 * np.pi * x) + rng3.normal(0, 0.3, 200)
    datasets['gaussian'] = pd.DataFrame({'x': x, 'y': y})

    # CR
    rng4 = np.random.default_rng(RNG_SEED)
    x = np.linspace(0, 1, 200)
    y = np.sin(2 * np.pi * x) + 0.4 * rng4.normal(0, 1, 200)
    datasets['cr'] = pd.DataFrame({'x': x, 'y': y})

    return datasets


def run_r(datasets):
    csvs = {}
    for k, df in datasets.items():
        f = tempfile.NamedTemporaryFile(suffix=f'_{k}.csv', delete=False, mode='w', newline='')
        df.to_csv(f, index=False)
        f.close()
        csvs[k] = f.name.replace('\\', '/')

    r_code = f"""
suppressMessages({{library(mgcv)}})

show <- function(name, m) {{
  cat(sprintf("=== %s ===\\n", name))
  cat(sprintf("  sp    = %.6f\\n", m$sp))
  cat(sprintf("  edf   = %.4f\\n", sum(m$edf)))
  cat(sprintf("  deviance = %.6f\\n", m$deviance))
  cat(sprintf("  fitted[1] = %.6f\\n", fitted(m)[1]))
  cat(sprintf("  fitted[100] = %.6f\\n", fitted(m)[100]))
  cat(sprintf("  coef[1] = %.6f\\n", coef(m)[1]))
}}

d <- read.csv("{csvs['poisson']}")
m <- gam(y ~ s(x), family=poisson(), data=d, method="REML")
show("poisson", m)

d <- read.csv("{csvs['binomial']}")
m <- gam(y ~ s(x), family=binomial(), data=d, method="REML")
show("binomial", m)

d <- read.csv("{csvs['gaussian']}")
m <- gam(y ~ s(x), data=d, method="REML")
show("gaussian", m)

d <- read.csv("{csvs['cr']}")
m <- gam(y ~ s(x, bs='cr'), data=d, method="REML")
show("cr", m)
"""
    rfile = tempfile.NamedTemporaryFile(suffix='.R', mode='w', delete=False)
    rfile.write(r_code)
    rfile.close()

    proc = subprocess.run([R_EXE, '--vanilla', rfile.name],
                          capture_output=True, text=True, timeout=120)
    os.unlink(rfile.name)
    for p in csvs.values():
        os.unlink(p)

    if proc.returncode != 0:
        print("R ERROR:", proc.stderr[-1000:])
    return proc.stdout


def run_py(datasets):
    from pymgcv.api.gam import GAM
    results = {}
    specs = {
        'poisson':  ('y ~ s(x)',          'poisson'),
        'binomial': ('y ~ s(x)',          'binomial'),
        'gaussian': ('y ~ s(x)',          'gaussian'),
        'cr':       ("y ~ s(x, bs='cr')", 'gaussian'),
    }
    for name, (formula, fam) in specs.items():
        df = datasets[name]
        m = GAM(formula, data=df, family=fam, method='REML')
        m.fit()
        mu = m.family.linkinv(m._X_fit @ m.beta)
        results[name] = {
            'sp': m.smoothing_parameters[0],
            'edf': m.edf,
            'fitted_1': float(mu[0]),
            'fitted_100': float(mu[99]),
            'coef_1': float(m.beta[0]),
        }
    return results


if __name__ == '__main__':
    datasets = make_datasets()
    print("--- R (mgcv) ---")
    print(run_r(datasets))
    print("--- Python (pymgcv) ---")
    py = run_py(datasets)
    for name, v in py.items():
        print(f"=== {name} ===")
        for k2, val in v.items():
            print(f"  {k2} = {val:.6f}")
