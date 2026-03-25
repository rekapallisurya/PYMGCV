# %%
from pymgcv import GAM, Tweedie
import pandas as pd

df = pd.read_csv("campaign_data.csv")

# Scale income to thousands — avoids numerical issues with large predictor values
df["income_k"] = df["income"] / 1000

# %%
model = GAM(
    formula="spend ~ s(age) + s(income_k) + s(contacts)",
    family=Tweedie(estimate_power=True),  # estimates p like R's tw()
    method="REML",
    control={"maxit": 50, "epsilon": 1e-4},
)

model.fit(df)
# %%
print(model.summary())
# %%
# Predictions
df["pred_mgcv"] = model.predict(df, scale="response")
# %%
# Save predictions
df.to_csv("mgcv_output.csv", index=False)
# %%
