"""Local polytropic-index calculations."""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - minimal/broken local environments.
    savgol_filter = None


def _smooth_log(values: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    log_values = np.log(np.clip(values, 1e-300, None))
    if len(log_values) < 5:
        return log_values
    window = min(window_length, len(log_values) if len(log_values) % 2 == 1 else len(log_values) - 1)
    if window <= polyorder:
        window = polyorder + 2 + ((polyorder + 2) % 2 == 0)
    if window > len(log_values):
        return log_values
    if savgol_filter is not None:
        return savgol_filter(log_values, window_length=window, polyorder=min(polyorder, window - 2), mode="interp")
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(log_values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def calculate_local_n(P_array: np.ndarray, rho_array: np.ndarray, r_array: np.ndarray) -> np.ndarray:
    """Calculate n(r) = (d ln P / d ln rho - 1)^-1 with smoothing."""

    pressure = np.asarray(P_array, dtype=float)
    rho = np.asarray(rho_array, dtype=float)
    radius = np.asarray(r_array, dtype=float)
    if not (len(pressure) == len(rho) == len(radius)):
        raise ValueError("P_array, rho_array, and r_array must have the same length")
    if len(radius) < 5:
        raise ValueError("At least five radial samples are required")

    order = np.argsort(radius)
    pressure = np.clip(pressure[order], 1e-300, None)
    rho = np.clip(rho[order], 1e-300, None)
    radius = radius[order]

    ln_p = _smooth_log(pressure)
    ln_rho = _smooth_log(rho)
    dlnp_dr = np.gradient(ln_p, radius, edge_order=2)
    dlnrho_dr = np.gradient(ln_rho, radius, edge_order=2)

    gamma_local = np.divide(
        dlnp_dr,
        dlnrho_dr,
        out=np.full_like(dlnp_dr, np.nan),
        where=np.abs(dlnrho_dr) > 1e-14,
    )
    n_local = np.divide(
        1.0,
        gamma_local - 1.0,
        out=np.full_like(gamma_local, np.nan),
        where=np.abs(gamma_local - 1.0) > 1e-12,
    )

    finite = np.isfinite(n_local)
    if finite.any():
        fill_value = float(np.nanmedian(n_local[finite]))
        n_local = np.where(finite, n_local, fill_value)
    else:
        n_local = np.full_like(radius, 1.5)

    n_local = np.clip(n_local, -25.0, 25.0)

    if len(radius) >= 12:
        fit_slice = slice(5, min(16, len(radius)))
        coeff = np.polyfit(radius[fit_slice], n_local[fit_slice], deg=2)
        n_local[:5] = np.polyval(coeff, radius[:5])

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))
    return n_local[inverse_order]
