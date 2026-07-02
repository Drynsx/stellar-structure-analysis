"""Physical deviation drivers for departures from simple polytropes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stellar_analyzer.core.constants import HBAR_CGS, K_B_CGS, M_E_CGS, M_P_CGS


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    integrator = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrator(y, x))


@dataclass
class ResidualResult:
    delta_n_obs: float
    delta_n_theory: float
    delta_global: float
    status: str

    def __float__(self) -> float:
        return self.delta_global


def calculate_delta_n_rad(beta: np.ndarray | float, T: np.ndarray | float, rho: np.ndarray | float) -> np.ndarray:
    beta_arr = np.clip(np.asarray(beta, dtype=float), 1e-6, 1.0)
    T_arr = np.clip(np.asarray(T, dtype=float), 1.0, None)
    rho_arr = np.clip(np.asarray(rho, dtype=float), 1e-99, None)
    correction = 1.0 + 0.1 * np.log1p(T_arr / 1e6) - 0.05 * np.log1p(rho_arr / 1e3)
    return 1.5 * ((1.0 - beta_arr) / beta_arr) * correction


def calculate_delta_n_mu(mu_array: np.ndarray, r_array: np.ndarray, H_rho: np.ndarray | float) -> np.ndarray:
    mu = np.clip(np.asarray(mu_array, dtype=float), 1e-12, None)
    radius = np.asarray(r_array, dtype=float)
    r_safe = np.maximum(radius, np.nanmax(radius) * 1e-9)
    dlnmu_dlnr = np.gradient(np.log(mu), np.log(r_safe), edge_order=2)
    return 0.8 * np.abs(dlnmu_dlnr) * np.asarray(H_rho, dtype=float)


def calculate_delta_n_conv(
    grad: np.ndarray | float,
    grad_ad: np.ndarray | float,
    tau_conv: np.ndarray | float,
    tau_dyn: np.ndarray | float,
) -> np.ndarray:
    # Surface cells in young models can carry enormous formal radiative
    # gradients. They are outside the calibrated range of this perturbative
    # correction and otherwise dominate a mass-weighted global residual.
    delta = np.clip(np.asarray(grad, dtype=float) - np.asarray(grad_ad, dtype=float), 0.0, 1.0)
    ratio = np.clip(
        np.asarray(tau_conv, dtype=float) / np.maximum(np.asarray(tau_dyn, dtype=float), 1e-30),
        0.0,
        10.0,
    )
    return 1.2 * delta * ratio


def calculate_delta_n_nuc(epsilon_array: np.ndarray, rho_array: np.ndarray, r_array: np.ndarray) -> float:
    epsilon = np.clip(np.asarray(epsilon_array, dtype=float), 0.0, None)
    rho = np.clip(np.asarray(rho_array, dtype=float), 0.0, None)
    radius = np.asarray(r_array, dtype=float)
    radius_outer = max(float(np.nanmax(radius)), 1e-30)
    luminosity_density = epsilon * rho * 4.0 * np.pi * radius**2
    total = _integrate_trapezoid(luminosity_density, radius)
    core_mask = radius <= 0.1 * radius_outer
    core = _integrate_trapezoid(luminosity_density[core_mask], radius[core_mask]) if core_mask.sum() >= 2 else 0.0
    concentration = core / max(total, 1e-99)
    return 0.5 * concentration


def calculate_delta_n_deg(T_array: np.ndarray, rho_array: np.ndarray, X_fraction: np.ndarray | float) -> np.ndarray:
    temperature = np.clip(np.asarray(T_array, dtype=float), 1.0, None)
    rho = np.clip(np.asarray(rho_array, dtype=float), 1e-99, None)
    x = np.clip(np.asarray(X_fraction, dtype=float), 1e-6, 1.0)
    electron_density = rho * x / M_P_CGS
    fermi_temperature = (HBAR_CGS**2 / (2.0 * M_E_CGS * K_B_CGS)) * (3.0 * np.pi**2 * electron_density) ** (2.0 / 3.0)
    eta = fermi_temperature / temperature
    return np.select(
        [eta < 1.0, eta < 3.0],
        [0.3 * eta, 0.3 + 0.5 * (eta - 1.0)],
        default=1.5,
    )


def _weighted_mean(value: np.ndarray | float, weights: np.ndarray | None = None) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if weights is None:
        return float(np.nanmean(arr))
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    return float(np.nansum(arr * w) / max(np.nansum(w), 1e-99))


def calculate_global_residual(
    n_global_observed: float,
    n_base: float,
    delta_n_list: list[np.ndarray | float],
    weights: np.ndarray | None = None,
) -> ResidualResult:
    """Compare observed polytropic departure to summed physical drivers."""

    delta_n_theory = sum(_weighted_mean(delta, weights=weights) for delta in delta_n_list)
    delta_n_obs = float(n_global_observed - n_base)
    delta_global = float(delta_n_obs - delta_n_theory)
    return ResidualResult(
        delta_n_obs=delta_n_obs,
        delta_n_theory=float(delta_n_theory),
        delta_global=delta_global,
        status="ANOMALOUS" if delta_global > 0.1 else "NORMAL",
    )
