import numpy as np
import pandas as pd

np.random.seed(42)
n = 10000

age = np.random.uniform(18, 70, n)
income = np.random.lognormal(mean=10, sigma=0.5, size=n)
campaign_contacts = np.random.poisson(3, n)

# Nonlinear true function
mu = np.exp(
    0.02 * age +
    0.00001 * income +
    -0.1 * campaign_contacts +
    0.0000001 * income * age
)

# Tweedie-like generation (zero + gamma)
p_zero = np.exp(-mu / np.max(mu))
is_zero = np.random.binomial(1, p_zero)

spend = np.where(
    is_zero == 1,
    0,
    np.random.gamma(shape=2, scale=mu/2)
)

df = pd.DataFrame({
    "age": age,
    "income": income,
    "contacts": campaign_contacts,
    "spend": spend
})

df.to_csv("campaign_data.csv", index=False)