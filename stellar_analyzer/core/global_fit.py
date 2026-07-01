"""Global Lane-Emden polytropic fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # Prefer SciPy's Levenberg-Marquardt optimizer when available.
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - minimal/broken local environments.
    least_squares = None


@dataclass
class GlobalFitResult:
    n_global: float
    K: float
    alpha: float
    reduced_chi2: float
    success: bool
    message: str = ""

    def __iter__(self):
        yield self.n_global
        yield self.K
        yield self.alpha
        yield self.reduced_chi2


def _theta_power(theta: float, n_index: float) -> float:
    return float(np.power(max(theta, 0.0), n_index))


def _lane_emden_rhs(xi: float, theta: float, phi: float, n_index: float) -> tuple[float, float]:
    dtheta = phi
    if xi <= 1e-12:
        dphi = 0.0
    else:
        dphi = -_theta_power(theta, n_index) - (2.0 / xi) * phi
    return dtheta, dphi


def solve_lane_emden_rk4(
    n_index: float,
    xi_max: float = 20.0,
    step: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the Lane-Emden equation with a fixed-step fourth-order RK scheme."""

    if not np.isfinite(n_index) or n_index <= 0:
        raise ValueError("Lane-Emden index must be positive")

    xi0 = 1e-6
    theta0 = 1.0 - xi0**2 / 6.0 + n_index * xi0**4 / 120.0
    phi0 = -xi0 / 3.0 + n_index * xi0**3 / 30.0

    xi_values = [xi0]
    theta_values = [theta0]
    xi = xi0
    theta = theta0
    phi = phi0

    while xi < xi_max and theta > 0.0 and len(xi_values) < 200000:
        h = min(step, xi_max - xi)

        k1_t, k1_p = _lane_emden_rhs(xi, theta, phi, n_index)
        k2_t, k2_p = _lane_emden_rhs(xi + 0.5 * h, theta + 0.5 * h * k1_t, phi + 0.5 * h * k1_p, n_index)
        k3_t, k3_p = _lane_emden_rhs(xi + 0.5 * h, theta + 0.5 * h * k2_t, phi + 0.5 * h * k2_p, n_index)
        k4_t, k4_p = _lane_emden_rhs(xi + h, theta + h * k3_t, phi + h * k3_p, n_index)

        theta += (h / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
        phi += (h / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
        xi += h

        xi_values.append(xi)
        theta_values.append(theta)

    xi_arr = np.asarray(xi_values)
    theta_arr = np.asarray(theta_values)
    theta_arr = np.clip(theta_arr, 0.0, None)
    return xi_arr, theta_arr


def _predict_density(radius: np.ndarray, n_index: float, K: float, alpha: float) -> np.ndarray:
    xi_target = radius / max(alpha, 1e-30)
    xi_max = max(float(np.nanmax(xi_target)) * 1.05, 1.0)
    xi_grid, theta_grid = solve_lane_emden_rk4(n_index, xi_max=xi_max)
    theta = np.interp(xi_target, xi_grid, theta_grid, left=theta_grid[0], right=0.0)
    theta = np.clip(theta, 0.0, None)
    return K * np.power(theta, n_index)


def _fallback_grid_fit(radius: np.ndarray, rho: np.ndarray, sigma: np.ndarray, n_initial: float):
    best = {"score": float("inf"), "n": n_initial, "K": float(np.nanmax(rho)), "alpha": max(radius[-1] / 3.5, 1e-6)}
    n_grid = np.unique(np.concatenate([np.linspace(0.8, 4.5, 19), np.array([n_initial])]))
    alpha_grid = radius[-1] / np.linspace(1.8, 9.0, 28)
    weights = 1.0 / np.maximum(sigma, 1e-30) ** 2
    for n_index in n_grid:
        for alpha in alpha_grid:
            try:
                basis = _predict_density(radius, float(n_index), 1.0, float(alpha))
            except Exception:
                continue
            denom = float(np.sum(weights * basis**2))
            if denom <= 0:
                continue
            K = float(np.sum(weights * rho * basis) / denom)
            if K <= 0 or not np.isfinite(K):
                continue
            score = float(np.sum(((rho - K * basis) / sigma) ** 2))
            if score < best["score"]:
                best = {"score": score, "n": float(n_index), "K": K, "alpha": float(alpha)}
    return np.array([best["n"], np.log(best["K"]), np.log(best["alpha"])], dtype=float), "NumPy grid-search fallback"


def fit_global_polytrope(r_array: np.ndarray, rho_array: np.ndarray, Teff: float) -> GlobalFitResult:
    """Fit ``n``, central-density scale ``K``, and radial scale ``alpha``.

    The optimizer uses SciPy's Levenberg-Marquardt implementation. Positivity
    of ``K`` and ``alpha`` is enforced with logarithmic parameters.
    """

    radius = np.asarray(r_array, dtype=float)
    rho = np.asarray(rho_array, dtype=float)
    mask = np.isfinite(radius) & np.isfinite(rho) & (rho > 0)
    radius = radius[mask]
    rho = rho[mask]
    if radius.size < 8:
        raise ValueError("Global fit requires at least eight finite density samples")

    order = np.argsort(radius)
    radius = radius[order]
    rho = rho[order]
    radius = radius - radius[0]
    if radius[-1] <= 0:
        raise ValueError("Radius grid must span a positive interval")

    n_initial = 3.0 if Teff > 10000.0 else 1.5
    k_initial = max(float(np.nanmax(rho)), 1e-30)
    alpha_initial = max(radius[-1] / 3.5, radius[-1] * 1e-3)
    sigma = np.maximum(0.03 * np.abs(rho), np.nanmedian(np.abs(rho)) * 1e-6)

    def residual(params: np.ndarray) -> np.ndarray:
        n_index = float(params[0])
        K = float(np.exp(params[1]))
        alpha = float(np.exp(params[2]))
        if n_index <= 0.05 or n_index > 8.0 or not np.isfinite(K) or not np.isfinite(alpha):
            return np.full_like(rho, 1e6)
        try:
            pred = _predict_density(radius, n_index, K, alpha)
        except Exception:
            return np.full_like(rho, 1e6)
        return (rho - pred) / sigma

    x0 = np.array([n_initial, np.log(k_initial), np.log(alpha_initial)], dtype=float)
    if least_squares is not None:
        try:
            result = least_squares(residual, x0, method="lm", max_nfev=200)
        except ValueError:
            result = least_squares(residual, x0, method="trf", max_nfev=300)
        x_fit = result.x
        success = bool(result.success)
        message = str(result.message)
    else:
        x_fit, message = _fallback_grid_fit(radius, rho, sigma, n_initial)
        success = True

    n_fit = float(x_fit[0])
    K_fit = float(np.exp(x_fit[1]))
    alpha_fit = float(np.exp(x_fit[2]))
    chi2 = float(np.sum(residual(x_fit) ** 2) / max(radius.size - 3, 1))
    return GlobalFitResult(
        n_global=n_fit,
        K=K_fit,
        alpha=alpha_fit,
        reduced_chi2=chi2,
        success=success,
        message=message,
    )
