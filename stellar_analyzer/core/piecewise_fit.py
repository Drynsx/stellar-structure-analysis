"""Piecewise zonal polytropic fits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - minimal/broken local environments.
    minimize = None


@dataclass
class PiecewiseFitResult:
    n_core: float
    n_rad: float
    n_conv: float
    chi2_piecewise: float
    continuity_errors: dict[str, float] = field(default_factory=dict)
    beats_global: bool | None = None
    success: bool = True
    message: str = ""

    def __iter__(self):
        yield self.n_core
        yield self.n_rad
        yield self.n_conv
        yield self.chi2_piecewise


def _labels(radius_fraction: np.ndarray, grad_ad: np.ndarray, grad_rad: np.ndarray) -> np.ndarray:
    """Assign the Chapter 3 radial zones, confirming convection in the envelope."""
    labels = np.full(radius_fraction.shape, 1, dtype=int)
    labels[radius_fraction < 0.25] = 0
    labels[radius_fraction >= 0.70] = 2
    labels[(radius_fraction >= 0.70) & (grad_rad > grad_ad)] = 2
    return labels


def _n_from_gamma(gamma: float) -> float:
    if abs(gamma - 1.0) < 1e-10:
        return float("inf")
    return float(1.0 / (gamma - 1.0))


def fit_piecewise(
    r_array: np.ndarray,
    rho_array: np.ndarray,
    P_array: np.ndarray,
    grad_ad: np.ndarray,
    grad_rad: np.ndarray,
    global_chi2: float | None = None,
) -> PiecewiseFitResult:
    """Fit zonal polytropes subject to exact pressure-continuity constraints."""

    radius = np.asarray(r_array, dtype=float)
    rho = np.clip(np.asarray(rho_array, dtype=float), 1e-300, None)
    pressure = np.clip(np.asarray(P_array, dtype=float), 1e-300, None)
    grad_ad = np.asarray(grad_ad, dtype=float)
    grad_rad = np.asarray(grad_rad, dtype=float)

    if not (len(radius) == len(rho) == len(pressure) == len(grad_ad) == len(grad_rad)):
        raise ValueError("All piecewise-fit arrays must have the same length")
    if len(radius) < 12:
        raise ValueError("Piecewise fitting requires at least 12 radial samples")

    order = np.argsort(radius)
    radius = radius[order]
    rho = rho[order]
    pressure = pressure[order]
    grad_ad = grad_ad[order]
    grad_rad = grad_rad[order]
    rfrac = radius / max(float(np.nanmax(radius)), 1e-30)

    labels = _labels(rfrac, grad_ad, grad_rad)
    log_rho = np.log(rho)
    log_p = np.log(pressure)
    sigma = np.maximum(0.03 * np.abs(log_p), 1e-3)

    zone_masks = [labels == idx for idx in range(3)]
    for idx, mask in enumerate(zone_masks):
        if mask.sum() < 3:
            raise ValueError(f"Piecewise zone {idx} requires at least three samples")

    def residual(params: np.ndarray) -> np.ndarray:
        gammas = params[:3]
        log_ks = params[3:]
        pieces: list[np.ndarray] = []
        for idx, mask in enumerate(zone_masks):
            if mask.sum() == 0:
                pieces.append(np.array([0.0]))
                continue
            pred = log_ks[idx] + gammas[idx] * log_rho[mask]
            pieces.append((log_p[mask] - pred) / sigma[mask])

        return np.concatenate(pieces)

    def continuity(params: np.ndarray) -> np.ndarray:
        values = []
        for boundary, pair in ((0.25, (0, 1)), (0.70, (1, 2))):
            boundary_log_rho = float(np.interp(boundary, rfrac, log_rho))
            left = params[3 + pair[0]] + params[pair[0]] * boundary_log_rho
            right = params[3 + pair[1]] + params[pair[1]] * boundary_log_rho
            values.append(left - right)
        return np.asarray(values)

    initial_gamma = np.array([5.0 / 3.0, 4.0 / 3.0, 5.0 / 3.0])
    initial_log_k = np.array([
        float(np.nanmedian(log_p[mask] - initial_gamma[idx] * log_rho[mask])) if mask.any() else 0.0
        for idx, mask in enumerate(zone_masks)
    ])
    for boundary, left, right in ((0.25, 0, 1), (0.70, 1, 2)):
        boundary_log_rho = float(np.interp(boundary, rfrac, log_rho))
        initial_log_k[right] = initial_log_k[left] + (initial_gamma[left] - initial_gamma[right]) * boundary_log_rho
    x0 = np.concatenate([initial_gamma, initial_log_k])
    if minimize is None:
        raise ImportError("SciPy is required for equality-constrained piecewise fitting")
    result = minimize(
        lambda params: float(np.sum(residual(params) ** 2)),
        x0,
        method="SLSQP",
        constraints={"type": "eq", "fun": continuity},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    x_fit = np.asarray(result.x)
    constraint_error = float(np.max(np.abs(continuity(x_fit))))
    if not result.success or constraint_error > 1e-8:
        raise RuntimeError(f"Piecewise constrained fit failed: {result.message}; continuity error={constraint_error:.3e}")

    gammas = x_fit[:3]
    chi2 = float(np.sum(residual(x_fit) ** 2) / max(len(radius) - 6, 1))
    errors = {}
    for boundary, name, pair in ((0.25, "core_rad", (0, 1)), (0.70, "rad_conv", (1, 2))):
        boundary_log_rho = float(np.interp(boundary, rfrac, log_rho))
        left = x_fit[3 + pair[0]] + x_fit[pair[0]] * boundary_log_rho
        right = x_fit[3 + pair[1]] + x_fit[pair[1]] * boundary_log_rho
        scale = max(abs(float(np.exp(left))), abs(float(np.exp(right))), 1e-300)
        errors[name] = float(abs(np.exp(left) - np.exp(right)) / scale)

    return PiecewiseFitResult(
        n_core=_n_from_gamma(float(gammas[0])),
        n_rad=_n_from_gamma(float(gammas[1])),
        n_conv=_n_from_gamma(float(gammas[2])),
        chi2_piecewise=chi2,
        continuity_errors=errors,
        beats_global=None if global_chi2 is None else bool(chi2 < global_chi2),
        success=bool(result.success),
        message=str(result.message),
    )
