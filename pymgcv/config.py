"""Package-wide configuration constants for pymgcv."""

from __future__ import annotations

# Default basis dimension for smooth terms
DEFAULT_K: int = 10

# Default spline order (degree + 1; cubic = 4)
DEFAULT_ORDER: int = 4

# Convergence tolerance for PIRLS and MAGIC
TOLERANCE: float = 1e-7

# Maximum PIRLS iterations
MAX_PIRLS_ITER: int = 25

# Maximum MAGIC outer iterations
MAX_MAGIC_ITER: int = 10

# Numerical safety floor for variances and weights
VARIANCE_FLOOR: float = 1e-10

# Maximum absolute value for linear predictor (prevents link-function overflow)
ETA_MAX: float = 100.0
