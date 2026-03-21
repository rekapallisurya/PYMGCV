
# ============================================
# R Script: PyMGCV vs MGCV Comparison
# ============================================
# Save this as compare_with_mgcv.R
# Run in R with: source("compare_with_mgcv.R")

library(mgcv)
library(jsonlite)

# Set seed for reproducibility
set.seed(42)

# ============================================
# TEST 1: GAUSSIAN GAM
# ============================================
cat("\n=== TEST 1: Gaussian GAM ===\n")

n <- 100
x <- seq(0, 1, length.out=n)
y <- 2 + 3*x + 2*sin(4*pi*x) + rnorm(n, sd=0.3)
df_gauss <- data.frame(x=x, y=y)

model_gauss <- gam(y ~ s(x), family=gaussian(), data=df_gauss)
summary(model_gauss)

# Extract outputs
gauss_coef <- coef(model_gauss)
gauss_edf <- sum(model_gauss$edf)
gauss_aic <- AIC(model_gauss)
gauss_pred <- predict(model_gauss, type="response")

results_gauss <- list(
    family = "gaussian",
    coefficients = gauss_coef,
    total_edf = gauss_edf,
    aic = gauss_aic,
    predictions = gauss_pred,
    deviance = model_gauss$deviance
)

# ============================================
# TEST 2: POISSON GAM
# ============================================
cat("\n=== TEST 2: Poisson GAM ===\n")

x <- seq(0, 3, length.out=100)
eta <- 0.5 + 0.6*x
y <- rpois(100, exp(eta))
df_pois <- data.frame(x=x, y=y)

model_pois <- gam(y ~ s(x), family=poisson(), data=df_pois)
summary(model_pois)

pois_coef <- coef(model_pois)
pois_edf <- sum(model_pois$edf)
pois_aic <- AIC(model_pois)
pois_pred <- predict(model_pois, type="response")

results_pois <- list(
    family = "poisson",
    coefficients = pois_coef,
    total_edf = pois_edf,
    aic = pois_aic,
    predictions = pois_pred,
    deviance = model_pois$deviance
)

# ============================================
# TEST 3: BINOMIAL GAM
# ============================================
cat("\n=== TEST 3: Binomial GAM ===\n")

x <- seq(0, 1, length.out=80)
p <- plogis(0 + 2*sin(4*pi*x))  # logit scale
y <- rbinom(80, 1, p)
df_binom <- data.frame(x=x, y=y)

model_binom <- gam(y ~ s(x), family=binomial(), data=df_binom)
summary(model_binom)

binom_coef <- coef(model_binom)
binom_edf <- sum(model_binom$edf)
binom_aic <- AIC(model_binom)
binom_pred <- predict(model_binom, type="response")

results_binom <- list(
    family = "binomial",
    coefficients = binom_coef,
    total_edf = binom_edf,
    aic = binom_aic,
    predictions = binom_pred,
    deviance = model_binom$deviance
)

# ============================================
# TEST 4: MULTIVARIATE GAM
# ============================================
cat("\n=== TEST 4: Multivariate GAM (Two Smooths) ===\n")

n <- 100
x1 <- seq(0, 1, length.out=n)
x2 <- seq(0, 2, length.out=n)
y <- 1.2*sin(4*pi*x1) + 0.8*cos(3*pi*x2) + rnorm(n, sd=0.2)
df_multi <- data.frame(x1=x1, x2=x2, y=y)

model_multi <- gam(y ~ s(x1) + s(x2), family=gaussian(), data=df_multi)
summary(model_multi)

multi_coef <- coef(model_multi)
multi_edf <- sum(model_multi$edf)
multi_aic <- AIC(model_multi)
multi_pred <- predict(model_multi, type="response")

results_multi <- list(
    family = "gaussian_bivariate",
    coefficients = multi_coef,
    total_edf = multi_edf,
    aic = multi_aic,
    predictions = multi_pred,
    deviance = model_multi$deviance
)

# ============================================
# SAVE RESULTS TO JSON
# ============================================
all_results <- list(
    gaussian = results_gauss,
    poisson = results_pois,
    binomial = results_binom,
    multivariate = results_multi
)

# Write to JSON
json_output <- toJSON(all_results, pretty=TRUE)
writeLines(json_output, "mgcv_baseline_results.json")

cat("\nResults saved to: mgcv_baseline_results.json\n")
cat("\nSummary of R/MGCV Results:\n")
cat("  Gaussian GAM:    AIC =", results_gauss$aic, "  EDF =", results_gauss$total_edf, "\n")
cat("  Poisson GAM:     AIC =", results_pois$aic, "  EDF =", results_pois$total_edf, "\n")
cat("  Binomial GAM:    AIC =", results_binom$aic, "  EDF =", results_binom$total_edf, "\n")
cat("  Multivariate:    AIC =", results_multi$aic, "  EDF =", results_multi$total_edf, "\n")
