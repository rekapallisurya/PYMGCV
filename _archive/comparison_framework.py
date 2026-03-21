"""
Comprehensive PyMGCV vs MGCV Comparison Framework

This module provides tools to:
1. Generate standard test datasets
2. Compare pymgcv results with mgcv (R) baseline
3. Score component performance
4. Identify optimization gaps
5. Track numerical equivalence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json


class ComponentScorecard:
    """Tracks and scores pymgcv components vs mgcv baseline."""
    
    def __init__(self):
        """Initialize scorecard for all MGCV components."""
        self.scores = {}
        self.components = {
            'families': {
                'gaussian': 0,
                'poisson': 0,
                'binomial': 0,
                'gamma': 0,
                'tweedie': 0,
                'negative.binomial': 0,
                'inverse.gaussian': 0,
            },
            'smooth_bases': {
                'tprs': 0,
                'cubic': 0,
                'bspline': 0,
                'pspline': 0,
                'tensor': 0,
                'cyclic': 0,
            },
            'optimization': {
                'gcv': 0,
                'aic': 0,
                'magic': 0,
                'reml': 0,
            },
            'inference': {
                'summary': 0,
                'predictions': 0,
                'confidence_intervals': 0,
                'significance_tests': 0,
            },
            'diagnostics': {
                'residuals': 0,
                'gam_check': 0,
                'influence': 0,
                'concurvity': 0,
            },
            'specification': {
                'formula_parsing': 0,
                'by_variables': 0,
                'weights': 0,
                'offset': 0,
            }
        }

    def score_component(self, category: str, component: str, score: float):
        """Record score for a component (0-100)."""
        if category not in self.components:
            raise ValueError(f"Unknown category: {category}")
        if component not in self.components[category]:
            raise ValueError(f"Unknown component: {component}")
        self.components[category][component] = float(np.clip(score, 0, 100))

    def get_category_score(self, category: str) -> float:
        """Get average score for a category."""
        scores = list(self.components[category].values())
        return float(np.mean(scores)) if scores else 0

    def get_overall_score(self) -> float:
        """Get overall PyMGCV score (0-100)."""
        all_scores = []
        for category in self.components.values():
            all_scores.extend(category.values())
        return float(np.mean(all_scores)) if all_scores else 0

    def summary(self) -> str:
        """Generate summary report."""
        lines = []
        lines.append("\n" + "="*70)
        lines.append("PYMGCV vs MGCV COMPONENT SCORECARD")
        lines.append("="*70)
        
        for category, components in self.components.items():
            cat_score = self.get_category_score(category)
            lines.append(f"\n{category.upper()}: {cat_score:.1f}/100")
            lines.append("-" * 50)
            for comp, score in components.items():
                status = "[+]" if score >= 80 else "[~]" if score >= 50 else "[-]"
                lines.append(f"  {status} {comp:20s}: {score:6.1f}/100")
        
        overall = self.get_overall_score()
        lines.append(f"\n{'='*70}")
        lines.append(f"OVERALL PYMGCV PARITY: {overall:.1f}/100")
        lines.append(f"{'='*70}\n")
        
        return "\n".join(lines)


class TestDataGenerator:
    """Generate standard test datasets for comparison."""
    
    @staticmethod
    def gaussian_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
        """Gaussian regression data."""
        np.random.seed(seed)
        x = np.linspace(0, 1, n)
        y = 2 + 3*x + 2*np.sin(4*np.pi*x) + np.random.normal(0, 0.3, n)
        return pd.DataFrame({'x': x, 'y': y})
    
    @staticmethod
    def poisson_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
        """Poisson count data."""
        np.random.seed(seed)
        x = np.linspace(0, 3, n)
        eta_true = 0.5 + 0.6*x
        mu = np.exp(eta_true)
        y = np.random.poisson(mu)
        return pd.DataFrame({'x': x, 'y': y})
    
    @staticmethod
    def binomial_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
        """Binomial binary data."""
        np.random.seed(seed)
        x = np.linspace(0, 1, n)
        p_true = 0.5 + 0.3*np.sin(6*np.pi*x)
        p_true = np.clip(p_true, 0.01, 0.99)
        y = np.random.binomial(1, p_true)
        return pd.DataFrame({'x': x, 'y': y})
    
    @staticmethod
    def gamma_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
        """Gamma positive-valued data."""
        np.random.seed(seed)
        x = np.linspace(0, 1, n)
        mu_true = np.exp(1.5 + 1.2*x)
        shape = 2.0
        y = np.random.gamma(shape, mu_true/shape)
        return pd.DataFrame({'x': x, 'y': y})
    
    @staticmethod
    def tweedie_data(n: int = 100, power: float = 1.5, seed: int = 42) -> pd.DataFrame:
        """Tweedie distributed data."""
        np.random.seed(seed)
        x = np.linspace(0, 1, n)
        mu_true = np.exp(1.2 + x)
        # Approximate with gamma (for Tweedie with 1 < p < 2)
        shape = 1.0 / power
        y = np.random.gamma(shape, mu_true/shape)
        return pd.DataFrame({'x': x, 'y': y})
    
    @staticmethod
    def multivariate_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
        """Data with multiple smooth terms."""
        np.random.seed(seed)
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(0, 2, n)
        y = (1.2*np.sin(4*np.pi*x1) + 
             0.8*np.cos(3*np.pi*x2) + 
             np.random.normal(0, 0.2, n))
        return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    
    @staticmethod
    def weighted_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
        """Data with heterogeneous variance."""
        np.random.seed(seed)
        x = np.linspace(0, 1, n)
        y = 2 + 3*x + np.random.normal(0, 0.2 + 0.5*x, n)
        weights = 1.0 / (0.2 + 0.5*x)**2  # Inverse variance weighting
        return pd.DataFrame({'x': x, 'y': y, 'weights': weights})


class ComparisonMetrics:
    """Compute metrics comparing PyMGCV vs MGCV outputs."""
    
    @staticmethod
    def coefficient_error(pymgcv_coef: np.ndarray, mgcv_coef: np.ndarray) -> Dict:
        """Compare coefficient vectors."""
        diff = pymgcv_coef - mgcv_coef
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        max_error = np.max(np.abs(diff))
        
        # Calculate relative error (handle near-zero coefficients)
        safe_mgcv = np.where(np.abs(mgcv_coef) < 1e-6, 1e-6, mgcv_coef)
        rel_error = np.mean(np.abs(diff / safe_mgcv))
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'max_error': float(max_error),
            'rel_error': float(rel_error),
        }
    
    @staticmethod
    def edf_error(pymgcv_edf: float, mgcv_edf: float) -> Dict:
        """Compare effective degrees of freedom."""
        abs_diff = abs(pymgcv_edf - mgcv_edf)
        rel_diff = abs_diff / max(mgcv_edf, 0.1) if mgcv_edf != 0 else abs_diff
        
        return {
            'abs_diff': float(abs_diff),
            'rel_diff': float(rel_diff),
            'pymgcv': float(pymgcv_edf),
            'mgcv': float(mgcv_edf),
        }
    
    @staticmethod
    def prediction_error(pymgcv_pred: np.ndarray, mgcv_pred: np.ndarray) -> Dict:
        """Compare predictions."""
        diff = pymgcv_pred - mgcv_pred
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        max_error = np.max(np.abs(diff))
        correlation = np.corrcoef(pymgcv_pred, mgcv_pred)[0, 1]
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'max_error': float(max_error),
            'correlation': float(correlation),
        }
    
    @staticmethod
    def criterion_error(pymgcv_value: float, mgcv_value: float, 
                       criterion: str = 'aic') -> Dict:
        """Compare AIC/GCV/etc."""
        abs_diff = abs(pymgcv_value - mgcv_value)
        rel_diff = abs_diff / max(abs(mgcv_value), 0.1)
        
        return {
            'pymgcv': float(pymgcv_value),
            'mgcv': float(mgcv_value),
            'abs_diff': float(abs_diff),
            'rel_diff': float(rel_diff),
        }
    
    @staticmethod
    def score_accuracy(pymgcv_metric: float, mgcv_metric: float,
                      tolerance: float = 1e-4, 
                      rel_tolerance: float = 0.05) -> float:
        """Score accuracy as 0-100 based on tolerance.
        
        100: Perfect match
        80: Within tolerance
        50: 10x tolerance
        0: >100x tolerance
        """
        abs_diff = abs(pymgcv_metric - mgcv_metric)
        safe_mgcv = max(abs(mgcv_metric), 1e-6)
        rel_diff = abs_diff / safe_mgcv
        
        # Check both absolute and relative tolerance
        if abs_diff <= tolerance and rel_diff <= rel_tolerance:
            return 100.0
        elif abs_diff <= 10*tolerance and rel_diff <= 10*rel_tolerance:
            return 80.0
        elif abs_diff <= 100*tolerance and rel_diff <= 100*rel_tolerance:
            return 50.0
        else:
            return 20.0


def create_r_comparison_script() -> str:
    """Generate R code for comparison with mgcv."""
    
    r_code = '''
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
cat("\\n=== TEST 1: Gaussian GAM ===\\n")

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
cat("\\n=== TEST 2: Poisson GAM ===\\n")

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
cat("\\n=== TEST 3: Binomial GAM ===\\n")

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
cat("\\n=== TEST 4: Multivariate GAM (Two Smooths) ===\\n")

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

cat("\\nResults saved to: mgcv_baseline_results.json\\n")
cat("\\nSummary of R/MGCV Results:\\n")
cat("  Gaussian GAM:    AIC =", results_gauss$aic, "  EDF =", results_gauss$total_edf, "\\n")
cat("  Poisson GAM:     AIC =", results_pois$aic, "  EDF =", results_pois$total_edf, "\\n")
cat("  Binomial GAM:    AIC =", results_binom$aic, "  EDF =", results_binom$total_edf, "\\n")
cat("  Multivariate:    AIC =", results_multi$aic, "  EDF =", results_multi$total_edf, "\\n")
'''
    
    return r_code


if __name__ == '__main__':
    # Example usage
    print("COMPARISON FRAMEWORK DEMONSTRATIONS\n")
    
    # 1. Generate test data
    print("="*70)
    print("GENERATING TEST DATASETS")
    print("="*70)
    
    gen = TestDataGenerator()
    
    datasets = {
        'Gaussian': gen.gaussian_data(),
        'Poisson': gen.poisson_data(),
        'Binomial': gen.binomial_data(),
        'Gamma': gen.gamma_data(),
        'Multivariate': gen.multivariate_data(),
    }
    
    for name, df in datasets.items():
        print(f"\n{name} Data:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First row: {df.iloc[0].to_dict()}")
    
    # 2. Initialize scorecard
    print("\n" + "="*70)
    print("INITIALIZING COMPONENT SCORECARD")
    print("="*70)
    
    scorecard = ComponentScorecard()
    
    # Simulate some scores (will be replaced with actual test results)
    scorecard.score_component('families', 'gaussian', 95)
    scorecard.score_component('families', 'poisson', 90)
    scorecard.score_component('families', 'binomial', 35)  # New, not fully working
    scorecard.score_component('families', 'gamma', 60)
    scorecard.score_component('families', 'tweedie', 50)
    scorecard.score_component('families', 'negative.binomial', 30)
    scorecard.score_component('families', 'inverse.gaussian', 30)
    
    scorecard.score_component('smooth_bases', 'tprs', 85)
    scorecard.score_component('smooth_bases', 'cubic', 40)
    scorecard.score_component('smooth_bases', 'bspline', 40)
    scorecard.score_component('smooth_bases', 'pspline', 40)
    scorecard.score_component('smooth_bases', 'tensor', 0)
    scorecard.score_component('smooth_bases', 'cyclic', 0)
    
    scorecard.score_component('optimization', 'gcv', 45)
    scorecard.score_component('optimization', 'aic', 30)
    scorecard.score_component('optimization', 'magic', 50)
    scorecard.score_component('optimization', 'reml', 30)
    
    print(scorecard.summary())
    
    # 3. Save R comparison script
    print("\n" + "="*70)
    print("R COMPARISON SCRIPT")
    print("="*70)
    
    r_script = create_r_comparison_script()
    with open('compare_with_mgcv.R', 'w') as f:
        f.write(r_script)
    
    print("Saved R script: compare_with_mgcv.R")
    print("\nTo use this script:")
    print("  1. Open R or RStudio")
    print("  2. Run: source('compare_with_mgcv.R')")
    print("  3. This generates: mgcv_baseline_results.json")
