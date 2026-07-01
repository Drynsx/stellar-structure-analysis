"""Compatibility exports for the consolidated Lane-Emden implementation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stellar_analyzer.core.global_fit import fit_global_polytrope, solve_lane_emden_rk4


def solve_le(n_val: float, xi_max: float = 20.0, step: float = 0.01):
    return solve_lane_emden_rk4(n_val, xi_max=xi_max, step=step)


__all__ = ["fit_global_polytrope", "solve_le", "solve_lane_emden_rk4"]
