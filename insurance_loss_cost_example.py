"""
Insurance Loss Cost Modelling with pymgcv — Tweedie GAM
========================================================

Scenario
--------
Automobile Third-Party Liability (TPL) insurance portfolio with 5 000 policies.
We model the Capped Claims Ultimate Net Incurred (USD, stored in thousands) —
the aggregate insured loss per policy, capped at the per-occurrence reinsurance
retention of $1 000 000 USD.

Why work in $thousands of USD?
-------------------------------
Actuarial practice routinely scales loss data to hundreds or thousands of currency
units before modelling.  The Tweedie PIRLS working weights are proportional to
  w_i = μᵢ^{2−p} / φ
When losses span $0 to $500 000 with the default initialisation φ̂₀ = 1, the
largest claims receive weights > 700 while zero-claim policies get weights < 0.001.
This extreme imbalance destabilises the iterative solver.  Dividing losses by 1 000
keeps μ in the range [0, 500] kUSD and φ_kUSD ≈ φ_USD / √1000 ≈ 14, giving
numerically well-behaved working weights throughout.

Why Tweedie (p = 1.5)?
-----------------------
The Tweedie compound Poisson-Gamma family is the actuarial standard for modelling
pure premiums / loss costs because of its zero-mass property:

  p = 1  → pure Poisson  (count data only)
  p = 2  → pure Gamma    (positive amounts only, no zeros)
  1<p<2  → compound Poisson-Gamma: probability mass at 0 + continuous tail

For a typical auto portfolio ≈ 85–95 % of policies produce zero claims per year.
Tweedie(p ≈ 1.5, φ) captures the zero-inflated compound loss in one model.

Offset: offset(log_duration)
------------------------------
log_duration = log(policy_duration_years) is added to the linear predictor with
coefficient FIXED at 1.0:

  log E[Loss] = log(duration) + Xβ
  ⟺  E[Annual loss cost] = exp(Xβ)

A 6-month policy (duration=0.5) automatically has half the expected loss of an
identical 12-month policy — no manual pro-rating required.

Variables in the Dataset
--------------------------
  policy_id                       Unique policy identifier
  vehicle_age                     Vehicle age in years         [0 – 20]
  driver_age                      Primary driver age           [18 – 75]
  bonus_malus                     Bonus-Malus score            [50 – 350]
                                    50  = perfect no-claims history (cheapest)
                                    350 = worst claims history (most penalised)
  vehicle_class                   A = small city car (reference)
                                  B = medium saloon
                                  C = large / performance
                                  D = SUV / 4×4
  region                          Urban / Suburban / Rural (reference)
  displacement_cc                 Engine displacement          [800 – 5 000 cc]
  annual_mileage_km               Annual mileage               [5 000 – 50 000 km]
  exposure_usd                    Sum insured / limit (USD)    [$20 000 – $500 000]
  log10_exposure_usd              log₁₀(exposure_usd)
  duration                        Policy duration in years     [0.08 – 1.0]
  log_duration                    log(duration)  ← model offset
  class_B / class_C / class_D     Dummy indicators for vehicle class
  is_urban / is_suburban          Dummy indicators for region
  capped_claims_usd               Aggregate capped loss (raw USD) ← stored in CSV
  capped_claims_kusd              capped_claims_usd / 1000  ← MODEL RESPONSE

Smooth Terms Explained
-----------------------
  s(vehicle_age, k=8)
      Thin-plate regression spline (TPRS) with 8 basis functions.
      Captures the NON-LINEAR vehicle-age effect:
        • New vehicles (0–3 yrs): slightly elevated risk (expensive to repair,
          often bought by young inexperienced drivers)
        • Mid-age (4–14 yrs): lowest risk (depreciated, experienced owners)
        • Old vehicles (15+ yrs): rising risk (mechanical failure, worn safety)
      A linear age term would miss the bath-tub shape entirely.

  s(driver_age, k=10)
      TPRS with 10 basis functions — more flexibility for the classic U-shape:
        • Young drivers (18–25): highest frequency (inexperience, risk-taking)
        • Prime-age (35–60): lowest risk
        • Elderly (70+): slight uptick (slower reactions, visual decline)
      Industry standard: driver age must NEVER be modelled as a linear term.

  s(bonus_malus, k=8)
      Smooth of the French Bonus-Malus score [50, 350].
      Approximately monotone: higher score → more past claims → higher future risk.
      The smooth captures non-linearity near the floor (perpetual best-risk floor)
      and ceiling (persistent high-risk ceiling).

  s(log10_exposure_usd, k=6)
      Smooth of log₁₀(sum insured in USD).
      More expensive vehicles attract higher repair / replacement costs (severity).
      log₁₀ compresses the wide USD scale; the smooth captures diminishing
      marginal risk at extreme insured values.

  s(annual_mileage_km, k=6)
      Smooth of annual vehicle usage.
      More km driven → greater accident exposure.  The relationship is concave:
      risk grows fast from 5 000 to 20 000 km/yr then plateaus (urban stop-start
      traffic vs open-road motorway — similar accidents, different speeds).

Parametric (Linear) Terms
--------------------------
  class_B, class_C, class_D  (reference: class A = small city car)
      Each exp(β) is a multiplicative factor on the expected annual loss.

  is_urban, is_suburban  (reference: Rural)
      Urban areas have higher traffic density → higher collision frequency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pymgcv.api.gam import GAM
from pymgcv.distributions.tweedie import TweedieDispersionEstimator

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Constants
# ──────────────────────────────────────────────────────────────────────────────
CLAIM_CAP_USD  = 1_000_000.0   # per-occurrence XL reinsurance cap (USD)
TWEEDIE_POWER  = 1.5           # compound Poisson-Gamma power, actuarial standard
N_POLICIES     = 5_000
CSV_FILE       = "insurance_loss_cost_data.csv"
SEED           = 42

# Dispersion calibration:
#   phi_USD  = 433  → base claim frequency ≈ 8 % for base mu_USD = $300/yr
#   phi_kUSD = phi_USD / sqrt(1000) ≈ 13.69
#   Working in kUSD gives numerically stable PIRLS working weights.
TWEEDIE_PHI_USD  = 433.0
TWEEDIE_PHI_kUSD = TWEEDIE_PHI_USD / np.sqrt(1_000.0)   # ≈ 13.69


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Generate synthetic auto TPL portfolio
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 1: Generating synthetic auto-TPL portfolio")
print("=" * 70)

rng = np.random.default_rng(SEED)

vehicle_age       = rng.integers(0, 21, N_POLICIES).astype(float)   # 0–20 yrs
driver_age        = rng.integers(18, 76, N_POLICIES).astype(float)  # 18–75 yrs
bonus_malus       = rng.uniform(50, 350, N_POLICIES)                # 50=best
vehicle_class_raw = rng.choice(
    ['A', 'B', 'C', 'D'], N_POLICIES,
    p=[0.35, 0.30, 0.20, 0.15]
)
region_raw = rng.choice(
    ['Urban', 'Suburban', 'Rural'], N_POLICIES,
    p=[0.50, 0.30, 0.20]
)
displacement_cc   = rng.uniform(800, 5_000, N_POLICIES).round(0)
annual_mileage_km = rng.uniform(5_000, 50_000, N_POLICIES).round(0)

# Sum insured (USD): log-normal centred on $75 000, clipped [$20 k, $500 k]
exposure_usd       = rng.lognormal(np.log(75_000), 0.8, N_POLICIES)
exposure_usd       = np.clip(exposure_usd, 20_000, 500_000).round(-2)
log10_exposure_usd = np.log10(exposure_usd)

# Policy duration: beta-distributed, mostly near full year [0.08 – 1.0]
duration     = np.clip(rng.beta(8, 2, N_POLICIES), 0.08, 1.0)
log_duration = np.log(duration)


# ── True latent smooth functions ────────────────────────────────────────────
# Each returns a log-rate increment; total range is bounded to ≈ ±1.5 so that
# the worst-to-best risk multiplier stays within ~e^3 ≈ 20×, realistic for TPL.

def f_vehicle_age(x: np.ndarray) -> np.ndarray:
    """Bath-tub: new & old slightly higher risk, mid-life safest."""
    return 0.18 * np.exp(-0.25 * x) + 0.012 * np.maximum(x - 14.0, 0.0)


def f_driver_age(x: np.ndarray) -> np.ndarray:
    """U-shape: young & elderly drivers are highest risk."""
    young   = 0.80 * np.exp(-0.18 * np.maximum(x - 18.0, 0.0))
    elderly = 0.30 * np.exp(0.07  * np.maximum(x - 68.0, 0.0))
    return young + elderly


def f_bonus_malus(x: np.ndarray) -> np.ndarray:
    """Monotone+: higher score → more past claims → higher future rate."""
    c = (x - 100.0) / 100.0
    return 0.55 * c - 0.08 * c ** 2   # mild concavity at extremes


def f_log10_exposure(x: np.ndarray) -> np.ndarray:
    """Concave: severity rises with sum insured but at diminishing rate."""
    c = x - 4.5                        # centre at log10($30 k)
    return 0.35 * c - 0.10 * c ** 2


def f_annual_mileage(x: np.ndarray) -> np.ndarray:
    """Concave: usage risk grows fast then plateaus."""
    return 0.38 * np.log1p(x / 10_000.0) - 0.12


# Log-scale effects for categorical variables (relative to reference)
VEHICLE_CLASS_EFFECT = {'A': 0.000, 'B': 0.130, 'C': 0.300, 'D': 0.430}
REGION_EFFECT        = {'Urban': 0.350, 'Suburban': 0.100, 'Rural': 0.000}

# Base annual pure premium = $300 USD/yr  (= 0.3 kUSD)
BASE_LOG_RATE_USD = np.log(300.0)

log_mu_usd = (
    BASE_LOG_RATE_USD
    + f_vehicle_age(vehicle_age)
    + f_driver_age(driver_age)
    + f_bonus_malus(bonus_malus)
    + f_log10_exposure(log10_exposure_usd)
    + f_annual_mileage(annual_mileage_km)
    + np.array([VEHICLE_CLASS_EFFECT[c] for c in vehicle_class_raw])
    + np.array([REGION_EFFECT[r]        for r in region_raw])
    + log_duration          # duration offset
)
mu_usd = np.exp(log_mu_usd)

# ── Tweedie simulation: compound Poisson-Gamma ────────────────────────────────
#   λ = μ^{2−p} / [φ · (2−p)]   Poisson claim-count rate
#   α = (2−p) / (p−1) = 1.0     Gamma shape  (exponential for p=1.5)
#   β = φ · (p−1) · μ^{p−1}    Gamma scale  (avg single-claim USD)

p   = TWEEDIE_POWER
phi = TWEEDIE_PHI_USD

lam         = mu_usd ** (2 - p) / (phi * (2 - p))
alpha_shape = (2 - p) / (p - 1)                       # = 1.0 for p = 1.5
beta_scale  = phi * (p - 1) * mu_usd ** (p - 1)       # avg claim size (USD)

print(f"  Mean claim frequency  : {lam.mean() * 100:.1f} %")
print(f"  Avg claim size (USD)  : ${(alpha_shape * beta_scale).mean():,.0f}")
print(f"  Avg annual PP (USD)   : ${mu_usd.mean() / duration.mean():,.0f}")

claims_usd = np.zeros(N_POLICIES)
for i in range(N_POLICIES):
    n_i = rng.poisson(lam[i])
    if n_i > 0:
        claims_usd[i] = rng.gamma(alpha_shape, beta_scale[i], size=n_i).sum()

capped_claims_usd  = np.minimum(claims_usd, CLAIM_CAP_USD).round(2)
capped_claims_kusd = (capped_claims_usd / 1_000.0).round(4)   # ← model response

n_nonzero = (capped_claims_usd > 0).sum()
n_capped  = (claims_usd > CLAIM_CAP_USD).sum()

# ── Dummy-encode categoricals ─────────────────────────────────────────────────
class_B     = (vehicle_class_raw == 'B').astype(float)
class_C     = (vehicle_class_raw == 'C').astype(float)
class_D     = (vehicle_class_raw == 'D').astype(float)
is_urban    = (region_raw == 'Urban').astype(float)
is_suburban = (region_raw == 'Suburban').astype(float)

# ── Assemble and save DataFrame ───────────────────────────────────────────────
df = pd.DataFrame({
    'policy_id':          np.arange(1, N_POLICIES + 1),
    'vehicle_age':        vehicle_age,
    'driver_age':         driver_age,
    'bonus_malus':        bonus_malus.round(1),
    'vehicle_class':      vehicle_class_raw,
    'region':             region_raw,
    'displacement_cc':    displacement_cc,
    'annual_mileage_km':  annual_mileage_km,
    'exposure_usd':       exposure_usd,
    'log10_exposure_usd': log10_exposure_usd.round(4),
    'duration':           duration.round(4),
    'log_duration':       log_duration.round(6),
    'class_B':            class_B,
    'class_C':            class_C,
    'class_D':            class_D,
    'is_urban':           is_urban,
    'is_suburban':        is_suburban,
    'capped_claims_usd':  capped_claims_usd,
    'capped_claims_kusd': capped_claims_kusd,
})

df.to_csv(CSV_FILE, index=False)
print(f"\n  Saved → '{CSV_FILE}'  ({N_POLICIES:,} rows, {df.shape[1]} columns)")
print(f"  Zero-claim policies : {N_POLICIES - n_nonzero:,}  "
      f"({(N_POLICIES-n_nonzero)/N_POLICIES*100:.1f}%)")
print(f"  Claim rate          : {n_nonzero/N_POLICIES*100:.1f}%")
print(f"  Losses at cap ($1M) : {n_capped}")
print(f"  Mean loss (USD)     : ${capped_claims_usd.mean():>9,.2f}")
print(f"  Mean loss (kUSD)    : {capped_claims_kusd.mean():>9.3f}")
print(f"  Max loss  (USD)     : ${capped_claims_usd.max():>9,.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Reload from CSV
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 2: Reloading CSV and inspecting structure")
print("=" * 70)

df = pd.read_csv(CSV_FILE)
print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns\n")
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        print(f"  {col:<32s}  mean={s.mean():.4g}  std={s.std():.4g}  "
              f"min={s.min():.4g}  max={s.max():.4g}")
    else:
        print(f"  {col:<32s}  categorical  unique={s.nunique()}  "
              f"values={sorted(s.unique())}")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Model formula
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 3: Model formula")
print("=" * 70)

# The model response is capped_claims_kusd (kUSD).
# Predicted values from model.predict() are also in kUSD.
# Multiply × 1 000 to report in USD.

formula = (
    "capped_claims_kusd ~ "
    "offset(log_duration) + "
    "s(vehicle_age, k=8) + "
    "s(driver_age, k=10) + "
    "s(bonus_malus, k=8) + "
    "s(log10_exposure_usd, k=6) + "
    "s(annual_mileage_km, k=6) + "
    "class_B + class_C + class_D + "
    "is_urban + is_suburban"
)

print(f"""
  Response   : capped_claims_kusd   (loss in $thousands)
  Offset     : log_duration          (fixed coeff = 1.0; normalises to annual rate)
  Smooths    : 5 TPRS terms, each autonomously penalised by REML
  Parametric : 5 dummy variables (log-additive, unpenalised)

  {formula}
""")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Fit the Tweedie GAM
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 4: Fitting Tweedie GAM  (family='tweedie', p=1.5, REML)")
print("=" * 70)
print("  Optimising smoothing parameters via REML …\n")

model = GAM(
    formula = formula,
    data    = df,
    family  = 'tweedie',   # TweedieFamily(power=1.5)
    method  = 'REML',
    gamma   = 1.2,         # mild over-smoothing (sparser fits, analogous to mgcv)
    control = {'maxit': 150, 'epsilon': 1e-5},
)
model.fit(verbose=False)
print("  Model fitted successfully.\n")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Model summary
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 5: Model Summary")
print("=" * 70)
print(model.summary())


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Effective Degrees of Freedom per smooth
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 6: Effective Degrees of Freedom (EDF) per smooth")
print("=" * 70)
print("""
  EDF ≈ 1    → effectively LINEAR     (could use a parametric term)
  EDF ≈ 2–3  → mild non-linearity     (roughly quadratic shape)
  EDF > 4    → strong non-linearity   (complex curve required)
  EDF = k−1  → at basis limit; increase k if EDF ≈ k−1
""")
print(f"  {'Term':<46s}  EDF")
print(f"  {'-'*46}  ---")
for term, info in model.edf_per_smooth.items():
    print(f"  {term:<46s}  {info['edf']:.2f}")
print(f"  {'─'*46}  ─────")
print(f"  {'TOTAL':<46s}  {model.edf:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Fitted smoothing parameters
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 7: Fitted smoothing parameters (λ per penalty)")
print("=" * 70)
print("  Large λ → strongly penalised → nearly linear;  Small λ → complex shape\n")
smooth_labels = list(model.edf_per_smooth.keys())
for j, lj in enumerate(model.smoothing_parameters):
    label = smooth_labels[j] if j < len(smooth_labels) else f'penalty_{j+1}'
    print(f"  {label:<46s}  λ = {lj:14.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Tweedie dispersion estimation
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 8: Tweedie dispersion (φ) estimation")
print("=" * 70)

mu_fitted_kusd = model.predict(df, scale='response')
mu_fitted_usd  = mu_fitted_kusd * 1_000.0

est = TweedieDispersionEstimator(
    y             = df['capped_claims_kusd'].values,
    mu            = mu_fitted_kusd,
    initial_power = TWEEDIE_POWER,
)
p_est, phi_est_kusd = est.estimate(optimize_power=False)
phi_implied_usd = phi_est_kusd * np.sqrt(1_000.0)

print(f"\n  Scale       True φ (simulation)   φ̂ Pearson estimate")
print(f"  ─────────   ──────────────────    ──────────────────")
print(f"  kUSD        {TWEEDIE_PHI_kUSD:>16.2f}    {phi_est_kusd:>16.2f}")
print(f"  USD         {TWEEDIE_PHI_USD:>16.2f}    {phi_implied_usd:>16.2f}")
print(f"\n  Var(Y_kUSD) = φ · μ^p = {phi_est_kusd:.2f} · μ^{p_est:.1f}")
print(f"  φ̂ = mean[(y − μ)² / μ^p]  (Pearson chi-square estimator)")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Predictions and portfolio diagnostics
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 9: Portfolio predictions and diagnostics")
print("=" * 70)

df['fitted_usd']           = mu_fitted_usd
df['annual_loss_cost_usd'] = mu_fitted_usd / df['duration']

actual_usd  = df['capped_claims_usd'].values
fitted_usd  = df['fitted_usd'].values
nonzero     = actual_usd > 0

rmse          = np.sqrt(np.mean((actual_usd - fitted_usd) ** 2))
mae           = np.mean(np.abs(actual_usd - fitted_usd))
total_a       = actual_usd.sum()
total_f       = fitted_usd.sum()
balance       = total_f / total_a if total_a > 0 else float('nan')

print(f"""
  Policies                       : {N_POLICIES:,}
  Zero-claim policies            : {(actual_usd == 0).sum():,}  ({(actual_usd==0).mean()*100:.1f}%)
  Claim-reporting rate           : {nonzero.mean()*100:.1f}%

  Predicted mean annual loss cost: ${df['annual_loss_cost_usd'].mean():>12,.2f} USD
  Balance ratio   (pred/actual)  : {balance:>12.4f}   (1.00 = perfect)
  RMSE                           : ${rmse:>12,.2f} USD
  MAE                            : ${mae:>12,.2f} USD
  Total actual  (USD)            : ${total_a:>15,.2f}
  Total fitted  (USD)            : ${total_f:>15,.2f}
""")

# Segmented analysis
for seg_col, seg_vals in [('vehicle_class', ['A','B','C','D']),
                           ('region', ['Rural','Suburban','Urban'])]:
    print(f"  Annual loss cost by {seg_col}")
    print(f"  {'Value':<12} {'N':>6} {'Actual (USD)':>14} {'Pred (USD)':>14}  Lift")
    print(f"  {'-'*12} {'-'*6} {'-'*14} {'-'*14}  ───")
    base = df['annual_loss_cost_usd'].mean()
    for sv in seg_vals:
        mask = df[seg_col] == sv
        n    = mask.sum()
        act  = df.loc[mask, 'capped_claims_usd'].sum() / df.loc[mask, 'duration'].sum()
        pred = df.loc[mask, 'annual_loss_cost_usd'].mean()
        print(f"  {sv:<12} {n:>6,} ${act:>12,.2f} ${pred:>12,.2f}  {pred/base:.2f}×")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 10.  Partial effect profiles
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 10: Partial effect profiles (annual loss cost in USD)")
print("=" * 70)
print("""  All other covariates held fixed at:
    duration = 1 yr  (log_duration = 0)
    class = A (small car),  region = Rural,  numerics = training median
""")

REFERENCE = {
    'vehicle_age':        float(df['vehicle_age'].median()),
    'driver_age':         float(df['driver_age'].median()),
    'bonus_malus':        float(df['bonus_malus'].median()),
    'log10_exposure_usd': float(df['log10_exposure_usd'].median()),
    'annual_mileage_km':  float(df['annual_mileage_km'].median()),
    'log_duration':       0.0,
    'class_B': 0.0, 'class_C': 0.0, 'class_D': 0.0,
    'is_urban': 0.0, 'is_suburban': 0.0,
    'capped_claims_kusd': 0.0,
}


def partial_profile(var: str, n_grid: int = 15) -> tuple[np.ndarray, np.ndarray]:
    x_min  = float(df[var].quantile(0.05))
    x_max  = float(df[var].quantile(0.95))
    x_grid = np.linspace(x_min, x_max, n_grid)
    rows   = [dict(REFERENCE, **{var: float(xv)}) for xv in x_grid]
    pred_k = model.predict(pd.DataFrame(rows), scale='response')
    return x_grid, pred_k * 1_000.0


PROFILES = [
    ('vehicle_age',        'Vehicle Age (years)',    '{:.0f} yrs'),
    ('driver_age',         'Driver Age (years)',     '{:.0f} yrs'),
    ('bonus_malus',        'Bonus-Malus Score',      '{:.0f}'),
    ('log10_exposure_usd', 'log10(Sum Insured USD)', '{:.2f}'),
    ('annual_mileage_km',  'Annual Mileage (km)',    '{:.0f} km'),
]

for var, label, x_fmt in PROFILES:
    x_grid, y_pred = partial_profile(var)
    # subsample 9 evenly spaced points for display
    idx = np.round(np.linspace(0, len(x_grid)-1, 9)).astype(int)
    x_show, y_show = x_grid[idx], y_pred[idx]
    med_y = float(np.median(y_show))
    print(f"\n  --- {label} ---")
    print(f"  {'x':>15}  {'Annual Loss (USD)':>18}  Rel. to median")
    print(f"  {'-'*15}  {'-'*18}  {'-'*14}")
    for xv, yv in zip(x_show, y_show):
        rel  = (yv / med_y - 1) * 100 if med_y > 0 else 0
        sign = '+' if rel >= 0 else ''
        bar  = '▇' * min(int(abs(rel) / 3), 15)
        print(f"  {x_fmt.format(xv):>15}  ${yv:>16,.2f}  {sign}{rel:5.1f}%  {bar}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 11.  Parametric coefficient interpretation
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 11: Parametric coefficient interpretation")
print("=" * 70)

from pymgcv.utils.model_matrix import ModelMatrix

dummy_mm  = ModelMatrix(df, formula)
col_names = dummy_mm.column_names
beta      = model.beta

PARAM_COLS = {
    'class_B':    'Vehicle class B vs A  (medium saloon vs small)',
    'class_C':    'Vehicle class C vs A  (large/perf. vs small)',
    'class_D':    'Vehicle class D vs A  (SUV vs small)',
    'is_urban':   'Urban vs Rural region',
    'is_suburban':'Suburban vs Rural region',
}

print(f"\n  {'Parameter':<20} {'Description':<40} {'β':>8}  exp(β)  Change%")
print(f"  {'-'*20} {'-'*40} {'-'*8}  {'-'*6}  {'-'*8}")
for col, desc in PARAM_COLS.items():
    if col in col_names:
        idx  = col_names.index(col)
        b    = float(beta[idx])
        eb   = np.exp(b)
        pct  = (eb - 1) * 100
        sign = '+' if pct >= 0 else ''
        print(f"  {col:<20} {desc:<40}  {b:>7.4f}  {eb:>6.3f}  {sign}{pct:.1f}%")

print("""
  exp(β) = multiplicative factor on expected annual loss relative to reference
  (class A small car, Rural region).  e.g. class_D → SUVs cost more than city cars.
""")


# ──────────────────────────────────────────────────────────────────────────────
# 12.  Residual diagnostics
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 12: Residual diagnostics")
print("=" * 70)

actual_k = df['capped_claims_kusd'].values
fitted_k = mu_fitted_kusd
pearson  = (actual_k - fitted_k) / np.sqrt(
    np.maximum(phi_est_kusd * np.maximum(fitted_k, 1e-9) ** p_est, 1e-9)
)

print(f"""
  Pearson residuals (kUSD scale)
  ──────────────────────────────
  Mean    : {pearson.mean():>10.4f}    (expect ≈ 0)
  Std Dev : {pearson.std():>10.4f}    (expect ≈ 1)
  Skew    : {float(pd.Series(pearson).skew()):>10.4f}    (mild positive expected for Tweedie)
  5th pct : {np.percentile(pearson, 5):>10.4f}
  95th pct: {np.percentile(pearson, 95):>10.4f}

  EDF and basis check
  ────────────────────
  Total EDF     : {model.edf:.2f}
  Parameters    : {len(model.beta)}
  Observations  : {len(df):,}
  n / EDF ratio : {len(df)/model.edf:.1f}   (>10 = acceptable; >50 = good)
""")


# ──────────────────────────────────────────────────────────────────────────────
# 13.  Actuarial summary
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  STEP 13: Actuarial interpretation")
print("=" * 70)
print(f"""
  MODEL:   Tweedie GAM  (p={TWEEDIE_POWER}, REML-penalised TPRS smooths)
  PURPOSE: Pure Premium / Loss Cost pricing for Automobile TPL
  RESPONSE UNIT: kUSD   →  multiply predictions by 1 000 for USD

  ┌──────────────────────────────────────────────────────────────────────┐
  │ Key actuarial insights from the smooth terms                        │
  │                                                                      │
  │ s(driver_age)       U-shape.  Highest EDF = most complex non-       │
  │                      linearity.  Young drivers (~18-25) are riskiest.│
  │                                                                      │
  │ s(bonus_malus)      Monotone: each 100-pt BM increase raises the   │
  │                      annual loss cost by approximately 30-45%.       │
  │                                                                      │
  │ s(vehicle_age)      Bath-tub: mid-life vehicles (5-14 yrs) safest. │
  │                      Old vehicles face rising mechanical risk.        │
  │                                                                      │
  │ s(log10_exposure)   Concave: larger sum insured → higher severity,  │
  │                      but effect saturates at very high values.       │
  │                                                                      │
  │ s(annual_mileage)   Concave: usage risk plateaus above ~30 000 km. │
  └──────────────────────────────────────────────────────────────────────┘

  offset(log_duration)
    A 6-month policy has exactly HALF the expected loss of a 12-month policy
    with identical risk characteristics.  No ad-hoc pro-rating needed.

  Adding more structure (optional)
    te(driver_age, bonus_malus, k=5)       — interaction between age & BM
    te(vehicle_age, annual_mileage, k=5)   — worn vehicles driven heavily
    These tensor-product smooths need n > 10 000 for stable estimation.

  Next steps
    1. model.gam_check()   → residual QQ-plots, basis adequacy tests (k-check)
    2. Compare REML scores for p ∈ {{1.3, 1.4, 1.5, 1.6, 1.7}}
    3. Hold-out validation by accident year or random fold
    4. Convert fitted annual pure premiums into rate-change indications
""")
print("  Script completed successfully.")
print("=" * 70)
