"""Numerical core for stellar structure analysis."""

from stellar_analyzer.core.global_fit import fit_global_polytrope
from stellar_analyzer.core.local_fit import calculate_local_n
from stellar_analyzer.core.piecewise_fit import fit_piecewise

__all__ = ["fit_global_polytrope", "calculate_local_n", "fit_piecewise"]
