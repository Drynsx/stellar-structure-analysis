"""Piecewise zonal polytropic fits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - minimal/broken local environments.
    least_squares = None


@dataclass
class PiecewiseFitResult:
    n_core: float
    n_rad: float
    n_conv: float
    chi2_piecewise: float
    continuity_errors: dict[str, float] = field(default_factory=dict)
    beats_global: bool | None = None

    def __iter__(self):
        yield self.n_core
        yield self.n_rad
        yield self.n_conv
        yield self.chi2_piecewise


def _labels(radius_fraction: np.ndarray, grad_ad: np.ndarray, grad_rad: np.ndarray) -> np.ndarray:
    labels = np.full(radius_fraction.shape, 1, dtype=int)
    labels[radius_fraction < 0.25] = 0
    labels[radius_fraction >= 0.70] = 2
    labels[grad_rad > grad_ad] = 2
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
    """Fit core, radiative, and convective polytropes with continuity penalties."""

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
            fallback = [(rfrac < 0.25), ((rfrac >= 0.25) & (rfrac < 0.70)), (rfrac >= 0.70)][idx]
            zone_masks[idx] = fallback

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

        penalties = []
        for boundary, pair in ((0.25, (0, 1)), (0.70, (1, 2))):
            boundary_log_rho = float(np.interp(boundary, rfrac, log_rho))
            left = params[3 + pair[0]] + params[pair[0]] * boundary_log_rho
            right = params[3 + pair[1]] + params[pair[1]] * boundary_log_rho
            penalties.append(np.sqrt(50.0) * (left - right))
        pieces.append(np.asarray(penalties))
        return np.concatenate(pieces)

    initial_gamma = np.array([5.0 / 3.0, 4.0 / 3.0, 5.0 / 3.0])
    initial_log_k = np.array([
        float(np.nanmedian(log_p[mask] - initial_gamma[idx] * log_rho[mask])) if mask.any() else 0.0
        for idx, mask in enumerate(zone_masks)
    ])
    x0 = np.concatenate([initial_gamma, initial_log_k])
    if least_squares is not None:
        result = least_squares(residual, x0, method="lm", max_nfev=300)
        x_fit = result.x
    else:
        x_fit = x0.copy()
        for idx, mask in enumerate(zone_masks):
            if mask.sum() >= 2:
                coeff = np.polyfit(log_rho[mask], log_p[mask], deg=1)
                x_fit[idx] = coeff[0]
                x_fit[3 + idx] = coeff[1]

    gammas = x_fit[:3]
    chi2 = float(np.sum(residual(x_fit) ** 2) / max(len(radius) - 6, 1))
    errors = {}
    for boundary, name, pair in ((0.25, "core_rad", (0, 1)), (0.70, "rad_conv", (1, 2))):
        boundary_log_rho = float(np.interp(boundary, rfrac, log_rho))
        left = x_fit[3 + pair[0]] + x_fit[pair[0]] * boundary_log_rho
        right = x_fit[3 + pair[1]] + x_fit[pair[1]] * boundary_log_rho
        errors[name] = float(np.exp(left) - np.exp(right))

    return PiecewiseFitResult(
        n_core=_n_from_gamma(float(gammas[0])),
        n_rad=_n_from_gamma(float(gammas[1])),
        n_conv=_n_from_gamma(float(gammas[2])),
        chi2_piecewise=chi2,
        continuity_errors=errors,
        beats_global=None if global_chi2 is None else bool(chi2 < global_chi2),
    )
