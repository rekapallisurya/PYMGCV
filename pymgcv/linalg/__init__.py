"""Linear algebra utilities: decompositions, solvers, trace computations."""

from __future__ import annotations

from .native_engine import backend_info, col_squared_norms, solve_spd_fortran

__all__ = [
    "backend_info",
    "col_squared_norms",
    "solve_spd_fortran",
]
