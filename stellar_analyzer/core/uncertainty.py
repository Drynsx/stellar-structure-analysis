"""Bootstrap and propagated uncertainty calculations."""

from __future__ import annotations

import numpy as np

from stellar_analyzer.core.deviation_drivers import calculate_delta_n_rad
from stellar_analyzer.core.global_fit import fit_global_polytrope


def bootstrap_global_n(
    r_array: np.ndarray,
    rho_array: np.ndarray,
    Teff: float,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Bootstrap ``n_global`` by resampling radial-density pairs."""

    radius = np.asarray(r_array, dtype=float)
    rho = np.asarray(rho_array, dtype=float)
    if len(radius) != len(rho):
        raise ValueError("r_array and rho_array must have the same length")
    if len(radius) < 8:
        raise ValueError("Bootstrap requires at least eight radial-density pairs")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    rng = np.random.default_rng(random_state)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(radius), len(radius))
        order = np.argsort(radius[idx])
        try:
            fit = fit_global_polytrope(radius[idx][order], rho[idx][order], Teff)
            if np.isfinite(fit.n_global):
                samples.append(float(fit.n_global))
        except Exception:
            continue

    sample_arr = np.asarray(samples, dtype=float)
    if sample_arr.size == 0:
        return {
            "requested_resamples": n_bootstrap, "n_success": 0, "success_rate": 0.0,
            "valid": False, "seed": random_state, "sigma_n": float("nan"),
            "mean_n": float("nan"), "confidence_interval_95": [float("nan"), float("nan")],
            "samples": sample_arr,
        }
    success_rate = float(sample_arr.size / n_bootstrap)
    return {
        "requested_resamples": n_bootstrap,
        "sigma_n": float(np.nanstd(sample_arr, ddof=1)) if sample_arr.size > 1 else 0.0,
        "mean_n": float(np.nanmean(sample_arr)),
        "n_success": int(sample_arr.size),
        "success_rate": success_rate,
        "valid": bool(success_rate >= 0.90),
        "seed": random_state,
        "confidence_interval_95": np.nanpercentile(sample_arr, [2.5, 97.5]).tolist(),
        "samples": sample_arr,
    }


def propagate_delta_n_rad_error(
    beta: float,
    T: float,
    rho: float,
    sigma_beta: float,
    sigma_T: float,
    sigma_rho: float,
) -> dict[str, float]:
    """Propagate beta, temperature, and density errors with finite derivatives."""

    base = float(calculate_delta_n_rad(beta, T, rho))
    steps = {
        "beta": max(abs(beta) * 1e-5, 1e-8),
        "T": max(abs(T) * 1e-5, 1e-2),
        "rho": max(abs(rho) * 1e-5, 1e-12),
    }

    finite_beta = (
        float(calculate_delta_n_rad(beta + steps["beta"], T, rho))
        - float(calculate_delta_n_rad(beta - steps["beta"], T, rho))
    ) / (2.0 * steps["beta"])
    finite_T = (
        float(calculate_delta_n_rad(beta, T + steps["T"], rho))
        - float(calculate_delta_n_rad(beta, T - steps["T"], rho))
    ) / (2.0 * steps["T"])
    finite_rho = (
        float(calculate_delta_n_rad(beta, T, rho + steps["rho"]))
        - float(calculate_delta_n_rad(beta, T, rho - steps["rho"]))
    ) / (2.0 * steps["rho"])

    beta_safe = float(np.clip(beta, 1e-6, 1.0))
    T_safe = float(max(T, 1.0))
    rho_safe = float(max(rho, 1e-99))
    correction = 1.0 + 0.1 * np.log1p(T_safe / 1e6) - 0.05 * np.log1p(rho_safe / 1e3)
    ratio = (1.0 - beta_safe) / beta_safe
    d_beta = -1.5 * correction / beta_safe**2
    d_T = 1.5 * ratio * 0.1 / (T_safe + 1e6)
    d_rho = 1.5 * ratio * -0.05 / (rho_safe + 1e3)
    sigma = np.sqrt((d_beta * sigma_beta) ** 2 + (d_T * sigma_T) ** 2 + (d_rho * sigma_rho) ** 2)
    return {
        "delta_n_rad": base,
        "sigma_delta_n_rad": float(sigma),
        "partial_beta": float(d_beta),
        "partial_T": float(d_T),
        "partial_rho": float(d_rho),
        "finite_difference_partial_beta": float(finite_beta),
        "finite_difference_partial_T": float(finite_T),
        "finite_difference_partial_rho": float(finite_rho),
        "derivative_max_relative_error": float(max(
            abs(d_beta - finite_beta) / max(abs(d_beta), 1e-30),
            abs(d_T - finite_T) / max(abs(d_T), 1e-30),
            abs(d_rho - finite_rho) / max(abs(d_rho), 1e-30),
        )),
    }
