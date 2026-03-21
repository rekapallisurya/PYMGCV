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
        """Starting mu for Tweedie: (y + mean(y)) / 2, matching mgcv.

        Using max(y, 0.1) puts 80-90 % of zero-claim observations at μ=0.1,
        far below the sample mean.  This causes huge (y-μ)/μ working responses
        for positive observations in the first PIRLS iteration, leading to
        step-halving to a wrong local minimum.

        mgcv's identical initialisation (see family.R, tweedie entry):
            mustart <- (y + mean(y)) / 2
        ensures all starting μ values are within the same order of magnitude
        as the sample mean, giving PIRLS a numerically stable starting point.
        """
        y = np.asarray(y, dtype=np.float64)
        mu_start = (y + np.maximum(y.mean(), 1e-6)) / 2.0
        return np.maximum(mu_start, 1e-6)

    @staticmethod
    def _tweedie_log_wright(y_pos: np.ndarray, phi: float, p: float) -> np.ndarray:
        """Compute log W(y, phi, p) — the Tweedie normalising constant for y > 0.

        Implements the series from Dunn & Smyth (2005) "Series evaluation of
        Tweedie exponential dispersion model densities", Stat & Comput 15:267-280.

        The j-th log-weight is (Dunn & Smyth eq. 3):

            log t_j = (j·α − 1)·log y  − log Γ(j+1) − log Γ(j·α)
                      − j·log(φ·(2−p)) − j·α·log(φ·(p−1))

        where α = (2−p)/(p−1).  The sum is evaluated in log space via
        log-sum-exp for numerical stability.

        Args:
            y_pos: Strictly positive responses, shape (m,).
            phi:   Dispersion parameter φ > 0.
            p:     Tweedie power, 1 < p < 2.

        Returns:
            log W(y_i, phi, p) for each y_i, shape (m,).
        """
        from scipy.special import gammaln

        alpha       = (2.0 - p) / (p - 1.0)
        log_phi_2mp = np.log(phi * (2.0 - p))
        log_phi_pm1 = np.log(phi * (p - 1.0))

        y_pos = np.asarray(y_pos, dtype=np.float64)
        log_y = np.log(y_pos)

        # Upper Poisson rate (dominant term index) for the largest y.
        # lambda_i = y_i^{2-p} / (phi*(2-p)); series dominated around j ~ lambda_i.
        lambda_max = float(np.max(y_pos ** (2.0 - p))) / (phi * (2.0 - p))
        j_max = int(np.ceil(lambda_max + 10.0 * np.sqrt(max(lambda_max, 0.5)))) + 50
        j_max = min(max(j_max, 30), 500)

        j_arr          = np.arange(1, j_max + 1, dtype=np.float64)  # (J,)
        gammaln_jp1    = gammaln(j_arr + 1.0)     # log j!
        gammaln_jalpha = gammaln(j_arr * alpha)    # log Γ(j·α)

        # Constant part of log t_j (does not depend on y)
        j_const = (- j_arr * log_phi_2mp
                   - j_arr * alpha * log_phi_pm1
                   - gammaln_jp1
                   - gammaln_jalpha)               # (J,)

        # Full log t_j(y_i): shape (N, J)
        log_t = ((j_arr[np.newaxis, :] * alpha - 1.0) * log_y[:, np.newaxis]
                 + j_const[np.newaxis, :])

        # Numerically stable log-sum-exp over j axis
        max_lt = np.max(log_t, axis=1, keepdims=True)
        log_W  = max_lt[:, 0] + np.log(np.sum(np.exp(log_t - max_lt), axis=1))
        return log_W

    def loglik(
        self, y: np.ndarray, mu: np.ndarray, dispersion: float = 1.0
    ) -> float:
        """Log-likelihood for Tweedie distribution (exponential family form).

        Full density from Dunn & Smyth (2005) including Wright function W:

            For y_i > 0:
              ℓ_i = y_i·μ_i^{1−p}/((1−p)·φ) − μ_i^{2−p}/((2−p)·φ) + log W(y_i, φ, p)

            For y_i = 0:
              ℓ_i =                           − μ_i^{2−p}/((2−p)·φ)

        The Wright function W is the normalising constant from the compound
        Poisson-Gamma representation; it depends on y, φ, p but NOT on μ.
        Including it makes the absolute log-likelihood (and hence AIC) comparable
        with R's mgcv::gam().

        Gradient: dLL/dμ = (y − μ) / (φ · μ^p)   (unchanged — W is μ-free)
        """
        p   = self.power
        phi = np.maximum(float(dispersion), 1e-10)
        mu  = np.maximum(np.asarray(mu,  dtype=np.float64), 1e-10)
        y   = np.asarray(y, dtype=np.float64)   # do NOT clip zeros

        # Exponential-family kernel: [y·θ(μ) − b(θ(μ))] / φ
        term1 = y * mu ** (1 - p) / ((1 - p) * phi)   # 0 when y = 0
        term2 = mu ** (2 - p) / ((2 - p) * phi)
        kernel = term1 - term2

        # Add Wright function normalising constant for positive observations.
        # W is constant in μ, so it does not affect PIRLS convergence or the
        # REML smoothing-parameter gradient — only absolute AIC/BIC values.
        mask = y > 0
        if np.any(mask):
            kernel[mask] += self._tweedie_log_wright(y[mask], phi, p)

        return float(np.sum(kernel))

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
