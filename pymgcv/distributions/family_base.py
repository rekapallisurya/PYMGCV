"""Exponential family distributions for GAM models.

Implements distribution families with methods:
    - linkinv(eta): inverse link function μ = g⁻¹(η)
    - dmu_deta(eta): derivative dμ/dη
    - variance(mu): variance function Var(Y) = V(μ)
    - loglik(y, mu): log-likelihood

Supported families:
    - Gaussian: μ = η, Var = σ²
    - Poisson: μ = exp(η), Var = μ
    - Gamma: μ = 1/η, Var = μ²/shape
    - Tweedie: Var = φ μ^p (1 < p < 2)

References:
    - McCullagh, P. & Nelder, J. (1989): Generalized Linear Models
    - Dunn, P. K. & Smyth, G. K. (2005): Tweedie distributions

Module exports:
    - Family: abstract base class
    - GaussianFamily, PoissonFamily, GammaFamily, TweedieFamily
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import special


class Family(ABC):
    """Abstract base class for exponential family distributions.

    Subclasses must implement:
        - linkinv(eta): inverse link function
        - dmu_deta(eta): derivative dμ/dη
        - variance(mu, dispersion): variance function
        - loglik(y, mu, dispersion): log-likelihood
        - linkfun(mu): forward link function (default: override in subclass)
        - initialize(y): sensible starting mu for PIRLS
    """

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        """Forward link: η = g(μ). Default: identity."""
        return mu.copy()

    def initialize(self, y: np.ndarray) -> np.ndarray:
        """Return sensible starting μ for PIRLS. Default: y."""
        return y.copy()

    @abstractmethod
    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        """Inverse link function: μ = g⁻¹(η).

        Args:
            eta: Linear predictor, shape (n,).

        Returns:
            Mean μ, shape (n,).
        """
        pass

    @abstractmethod
    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """Derivative of inverse link: dμ/dη.

        Args:
            eta: Linear predictor, shape (n,).

        Returns:
            Derivative, shape (n,).
        """
        pass

    @abstractmethod
    def variance(self, mu: np.ndarray, dispersion: float = 1.0) -> np.ndarray:
        """Variance function: Var(Y) = V(μ).

        Args:
            mu: Mean vector, shape (n,).
            dispersion: Dispersion parameter φ.

        Returns:
            Variance, shape (n,).
        """
        pass

    @abstractmethod
    def loglik(
        self, y: np.ndarray, mu: np.ndarray, dispersion: float = 1.0
    ) -> float:
        """Log-likelihood for observations.

        Args:
            y: Observations, shape (n,).
            mu: Predicted mean, shape (n,).
            dispersion: Dispersion parameter φ.

        Returns:
            Log-likelihood (scalar).
        """
        pass


class GaussianFamily(Family):
    """Gaussian (normal) family.

    Identity link: η = μ
    Variance: Var(Y) = φ
    """

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        """Identity link: μ = η."""
        return eta

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """dμ/dη = 1."""
        return np.ones_like(eta)

    def variance(self, mu: np.ndarray, dispersion: float = 1.0) -> np.ndarray:
        """Variance = φ (constant)."""
        return np.full_like(mu, dispersion)

    def loglik(
        self, y: np.ndarray, mu: np.ndarray, dispersion: float = 1.0
    ) -> float:
        """Log-likelihood for Gaussian.

        LL = -1/2 Σ (y - μ)² / φ - (n/2) log(φ)
        """
        residuals = y - mu
        ssr = np.sum(residuals**2)
        return -0.5 * ssr / dispersion


class PoissonFamily(Family):
    """Poisson family.

    Log link: η = log(μ)
    Variance: Var(Y) = μ
    """

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        """Exponential link: μ = exp(η)."""
        return np.exp(eta)

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """dμ/dη = exp(η)."""
        return np.exp(eta)

    def variance(self, mu: np.ndarray, dispersion: float = 1.0) -> np.ndarray:
        """Variance = μ."""
        return mu

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        """Log link: η = log(μ)."""
        return np.log(np.maximum(mu, 1e-300))

    def initialize(self, y: np.ndarray) -> np.ndarray:
        """Starting mu for Poisson: y + 0.1 (avoids log(0))."""
        return np.maximum(y, 0.0) + 0.1

    def loglik(
        self, y: np.ndarray, mu: np.ndarray, dispersion: float = 1.0
    ) -> float:
        """Log-likelihood for Poisson.

        LL = Σ (y log(μ) - μ - log(y!))
        """
        # Poisson loglik: y log(mu) - mu (ignoring log(y!) constant)
        return np.sum(y * np.log(mu) - mu)


class GammaFamily(Family):
    """Gamma family.

    Log link: η = log(μ)
    Variance: Var(Y) = φ μ²
    """

    def __init__(self, shape: float = 1.0) -> None:
        """Initialize Gamma family.

        Args:
            shape: Shape parameter α (default 1.0 = exponential).
        """
        self.shape = float(shape)

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        """Exponential link: μ = exp(η)."""
        return np.exp(eta)

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """dμ/dη = exp(η)."""
        return np.exp(eta)

    def variance(self, mu: np.ndarray, dispersion: float = 1.0) -> np.ndarray:
        """Variance = φ μ²."""
        return dispersion * mu**2

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        """Log link: η = log(μ)."""
        return np.log(np.maximum(mu, 1e-300))

    def initialize(self, y: np.ndarray) -> np.ndarray:
        """Starting mu for Gamma: max(y, epsilon)."""
        return np.maximum(y, 1e-6)

    def loglik(
        self, y: np.ndarray, mu: np.ndarray, dispersion: float = 1.0
    ) -> float:
        """Log-likelihood for Gamma.

        Parameterized by mean μ and shape α.
        LL = Σ [α (log(α) - log(μ)) - α y/μ + (α-1) log(y) - log(Γ(α))]
        """
        alpha = 1.0 / dispersion
        loglik = np.sum(
            alpha * np.log(alpha / mu)
            - alpha * y / mu
            + (alpha - 1) * np.log(y)
            - special.loggamma(alpha)
        )
        return loglik


class TweedieFamily(Family):
    """Tweedie family.

    Log link: η = log(μ)
    Variance: Var(Y) = φ μ^p   (1 < p < 2)

    The Tweedie distribution includes Poisson (p=1), Gamma (p>2), and
    Poisson-Gamma compound (1 < p < 2).

    Attributes:
        power: Power parameter p ∈ (1, 2).
    """

    def __init__(self, power: float = 1.5) -> None:
        """Initialize Tweedie family.

        Args:
            power: Power parameter p. Must be 1 < p < 2 for compound Poisson.

        Raises:
            ValueError: If power not in valid range (typically 1 < p < 2).
        """
        if not (1.0 < power < 2.0):
            raise ValueError(f'Tweedie power must be in (1, 2), got {power}')
        self.power = float(power)

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        """Log link: μ = exp(η)."""
        return np.exp(eta)

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """dμ/dη = exp(η)."""
        return np.exp(eta)

    def variance(self, mu: np.ndarray, dispersion: float = 1.0) -> np.ndarray:
        """Variance = φ μ^p."""
        return dispersion * mu**self.power

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        """Log link: η = log(μ)."""
        return np.log(np.maximum(mu, 1e-300))

    def initialize(self, y: np.ndarray) -> np.ndarray:
        """Starting mu for Tweedie: max(y, 0.1)."""
        return np.maximum(y, 0.1)

    def loglik(
        self, y: np.ndarray, mu: np.ndarray, dispersion: float = 1.0
    ) -> float:
        """Log-likelihood for Tweedie distribution.

        Approximation for 1 < p < 2 (compound Poisson-Gamma).
        LL ≈ Σ [y / ((1-p) μ^(1-p)) - μ^(2-p) / ((2-p) φ)]

        Note: Exact Tweedie loglik is complex; this is a standard approximation.
        """
        p = self.power
        phi = dispersion
        
        # Ensure numerical stability
        mu = np.maximum(mu, 1e-10)
        y = np.maximum(y, 1e-10)
        
        # Tweedie loglik terms
        term1 = y / ((1 - p) * mu**(1 - p))
        term2 = mu**(2 - p) / ((2 - p) * phi)
        
        return np.sum(term1 - term2)

    def summary(self) -> str:
        """Return summary of Tweedie family."""
        return f'TweedieFamily(power={self.power})'


class BinomialFamily(Family):
    """Binomial family for binary/proportion responses.

    Logit link (default): η = logit(μ) = log(μ/(1-μ))
    Variance: Var(Y) = μ(1-μ)
    
    Supports multiple link functions:
    - logit: η = log(μ/(1-μ))
    - probit: η = Φ⁻¹(μ)
    - cloglog: η = log(-log(1-μ))

    Attributes:
        link: Link function ('logit', 'probit', 'cloglog'). Default 'logit'.
    """

    def __init__(self, link: str = 'logit') -> None:
        """Initialize Binomial family.

        Args:
            link: Link function ('logit', 'probit', 'cloglog'). Default 'logit'.

        Raises:
            ValueError: If link not in supported options.
        """
        if link not in ('logit', 'probit', 'cloglog'):
            raise ValueError(f'Unsupported link: {link}. Choose from: logit, probit, cloglog')
        self.link = link

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        """Inverse link function: μ = g⁻¹(η)."""
        if self.link == 'logit':
            # μ = 1/(1 + exp(-η)) = exp(η)/(1 + exp(η))
            # Use numerically stable version
            return 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        elif self.link == 'probit':
            # μ = Φ(η) (standard normal CDF)
            return special.ndtr(eta)
        elif self.link == 'cloglog':
            # μ = 1 - exp(-exp(η))
            return 1.0 - np.exp(-np.exp(np.clip(eta, -500, 500)))

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        """Forward link: η = g(μ)."""
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        if self.link == 'logit':
            return np.log(mu / (1 - mu))
        elif self.link == 'probit':
            return special.ndtri(mu)
        elif self.link == 'cloglog':
            return np.log(-np.log(1 - mu))
        return np.log(mu / (1 - mu))  # default to logit

    def initialize(self, y: np.ndarray) -> np.ndarray:
        """Starting mu for Binomial: (y + 0.5) / 2, clipped to (0.01, 0.99)."""
        return np.clip((y + 0.5) / 2.0, 0.01, 0.99)

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """Derivative of inverse link: dμ/dη."""
        if self.link == 'logit':
            # d/dη[1/(1+exp(-η))] = exp(-η)/(1+exp(-η))² = μ(1-μ)
            mu = self.linkinv(eta)
            return mu * (1 - mu)
        elif self.link == 'probit':
            # d/dη[Φ(η)] = φ(η) = standard normal PDF
            return np.exp(-0.5 * eta ** 2) / np.sqrt(2 * np.pi)
        elif self.link == 'cloglog':
            # d/dη[1 - exp(-exp(η))] = exp(η - exp(η))
            return np.exp(eta - np.exp(np.clip(eta, -500, 500)))

    def variance(self, mu: np.ndarray, dispersion: float = 1.0) -> np.ndarray:
        """Variance = μ(1-μ) [dispersion parameter ignored for binomial]."""
        return mu * (1 - mu)

    def loglik(
        self, y: np.ndarray, mu: np.ndarray, dispersion: float = 1.0
    ) -> float:
        """Log-likelihood for Binomial.

        LL = Σ [y log(μ) + (1-y) log(1-μ)]
        
        Assumes binary data (y ∈ {0, 1}).
        For grouped data, supply y = count of successes and weights = total counts.
        """
        # Clip mu to avoid log(0)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))

    def summary(self) -> str:
        """Return summary of Binomial family."""
        return f'BinomialFamily(link={self.link})'


class NegativeBinomialFamily(Family):
    """Negative Binomial family for overdispersed count data.

    Log link: η = log(μ)
    Variance: Var(Y) = μ + μ²/θ
    
    The negative binomial distribution is a generalization of Poisson for
    overdispersed count data. The parameter θ is the shape/dispersion parameter.

    Attributes:
        theta: Shape parameter θ > 0. Larger θ → closer to Poisson.
               Default 1.0 (moderate overdispersion).
    """

    def __init__(self, theta: float = 1.0) -> None:
        """Initialize Negative Binomial family.

        Args:
            theta: Shape parameter θ > 0. Default 1.0.

        Raises:
            ValueError: If theta <= 0.
        """
        if theta <= 0:
            raise ValueError(f'Theta must be positive, got {theta}')
        self.theta = float(theta)

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        """Log link: μ = exp(η)."""
        return np.exp(eta)

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """dμ/dη = exp(η)."""
        return np.exp(eta)

    def variance(self, mu: np.ndarray, dispersion: float = 1.0) -> np.ndarray:
        """Variance = μ + μ²/θ."""
        return mu + mu**2 / self.theta

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        """Log link: η = log(μ)."""
        return np.log(np.maximum(mu, 1e-300))

    def initialize(self, y: np.ndarray) -> np.ndarray:
        """Starting mu for NegBin: max(y, 0.1)."""
        return np.maximum(y, 0.0) + 0.1

    def loglik(
        self, y: np.ndarray, mu: np.ndarray, dispersion: float = 1.0
    ) -> float:
        """Log-likelihood for Negative Binomial.

        Parameterized by mean μ and shape θ:
        LL = Σ [log(Γ(y+θ)) - log(Γ(θ)) - log(y!) + θ log(θ) - (y+θ) log(μ+θ)]
        """
        theta = self.theta
        
        # Ensure numerical stability
        mu = np.maximum(mu, 1e-10)
        
        # Negative binomial loglik (using gamma functions)
        loglik = np.sum(
            special.loggamma(y + theta)
            - special.loggamma(theta)
            - special.loggamma(y + 1)
            + theta * np.log(theta)
            - (y + theta) * np.log(mu + theta)
        )
        return loglik

    def summary(self) -> str:
        """Return summary of Negative Binomial family."""
        return f'NegativeBinomialFamily(theta={self.theta})'


class InverseGaussianFamily(Family):
    """Inverse Gaussian family for heavy-tailed continuous data.

    1/μ² link: η = 1/μ²
    Variance: Var(Y) = φ μ³
    
    The Inverse Gaussian (also called Wald) distribution is useful for
    heavy-tailed positive continuous data like insurance claims, survival times.

    Attributes:
        link: Link function ('inverse-square' or '1/mu^2'). Default 'inverse-square'.
    """

    def __init__(self, link: str = 'inverse-square') -> None:
        """Initialize Inverse Gaussian family.

        Args:
            link: Link function. Default 'inverse-square' (1/μ²).

        Raises:
            ValueError: If link not supported.
        """
        if link not in ('inverse-square', '1/mu^2'):
            raise ValueError(f'Unsupported link: {link}. Choose: inverse-square or 1/mu^2')
        self.link = link

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        """Inverse link: μ = 1/√η (inverse of 1/μ²)."""
        # η = 1/μ² → μ = 1/√η
        # Clip eta to avoid numerical issues
        eta_safe = np.maximum(eta, 1e-10)
        return 1.0 / np.sqrt(eta_safe)

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        """Inverse square link: η = 1/μ²."""
        return 1.0 / np.maximum(mu, 1e-300) ** 2

    def initialize(self, y: np.ndarray) -> np.ndarray:
        """Starting mu for InverseGaussian: (y + mean(y)) / 2, strictly positive."""
        mu_bar = float(np.mean(y[y > 0])) if np.any(y > 0) else 1.0
        return np.maximum((y + mu_bar) / 2.0, 1e-6)

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        """Derivative: dμ/dη = -1/(2η^(3/2))."""
        eta_safe = np.maximum(eta, 1e-10)
        return -0.5 / (eta_safe**(1.5))

    def variance(self, mu: np.ndarray, dispersion: float = 1.0) -> np.ndarray:
        """Variance = φ μ³."""
        return dispersion * mu**3

    def loglik(
        self, y: np.ndarray, mu: np.ndarray, dispersion: float = 1.0
    ) -> float:
        """Log-likelihood for Inverse Gaussian.

        Parameterized by mean μ and dispersion φ:
        LL = -n/2 log(2πφ) - Σ[(y - μ)² / (2φ μ² y)]
        """
        phi = dispersion
        
        # Ensure numerical stability
        mu = np.maximum(mu, 1e-10)
        y = np.maximum(y, 1e-10)
        
        n = len(y)
        loglik = -0.5 * n * np.log(2 * np.pi * phi)
        loglik -= np.sum((y - mu)**2 / (2 * phi * mu**2 * y))
        
        return loglik

    def summary(self) -> str:
        """Return summary of Inverse Gaussian family."""
        return f'InverseGaussianFamily(link={self.link})'


class BetaFamily(Family):
    """Beta regression family for proportional responses in (0, 1).

    mgcv equivalent: betar()

    Parameterisation: Y ~ Beta(µφ, (1-µ)φ) where µ ∈ (0,1), φ > 0.
    Logit link: η = log(µ/(1-µ))
    Variance: V(µ) = µ(1-µ) / (1+φ)  (dispersion φ estimated from residuals)

    References:
        - Ferrari & Cribari-Neto (2004): Beta regression for modelling proportions.
        - Wood (2017) §7.2.
    """

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return np.log(mu / (1 - mu))

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        mu = self.linkinv(eta)
        return np.clip(mu * (1 - mu), 1e-15, None)

    def variance(self, mu: np.ndarray, dispersion: float = 1.0) -> np.ndarray:
        """V(µ) = µ(1-µ) / (1 + φ).  dispersion = φ."""
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return mu * (1 - mu) / (1.0 + max(dispersion, 1e-6))

    def initialize(self, y: np.ndarray) -> np.ndarray:
        return np.clip(y, 0.01, 0.99)

    def loglik(self, y: np.ndarray, mu: np.ndarray, dispersion: float = 1.0) -> float:
        """Beta log-likelihood.

        LL = Σ [ log Γ(φ) - log Γ(µφ) - log Γ((1-µ)φ)
                 + (µφ - 1) log y + ((1-µ)φ - 1) log(1-y) ]
        """
        phi = max(dispersion, 1e-6)
        mu  = np.clip(mu, 1e-10, 1 - 1e-10)
        y   = np.clip(y,  1e-10, 1 - 1e-10)
        a = mu * phi
        b = (1 - mu) * phi
        return float(np.sum(
            special.loggamma(phi)
            - special.loggamma(a)
            - special.loggamma(b)
            + (a - 1) * np.log(y)
            + (b - 1) * np.log(1 - y)
        ))

    def summary(self) -> str:
        return 'BetaFamily(link=logit)'


class GaulssFamily(Family):
    """Gaussian location-scale family (simplified single-predictor version).

    mgcv equivalent: gaulss()  — note: full gaulss models both µ and log(σ)
    with separate linear predictors.  This implementation fits µ via a standard
    Gaussian model and estimates σ² from (Pearson) residuals at each PIRLS step,
    matching the behaviour of a heteroscedastic Gaussian model with constant σ².

    For a full two-predictor gaulss (heteroscedastic smoothing), a separate
    architectural extension is required.
    """

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        return eta.copy()

    def dmu_deta(self, eta: np.ndarray) -> np.ndarray:
        return np.ones_like(eta)

    def variance(self, mu: np.ndarray, dispersion: float = 1.0) -> np.ndarray:
        return np.full_like(mu, max(dispersion, 1e-10))

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        return mu.copy()

    def initialize(self, y: np.ndarray) -> np.ndarray:
        return y.copy()

    def loglik(self, y: np.ndarray, mu: np.ndarray, dispersion: float = 1.0) -> float:
        phi = max(dispersion, 1e-10)
        resid = y - mu
        n = len(y)
        return float(-0.5 * (np.sum(resid ** 2) / phi + n * np.log(2 * np.pi * phi)))

    def summary(self) -> str:
        return 'GaulssFamily(link=identity)'
