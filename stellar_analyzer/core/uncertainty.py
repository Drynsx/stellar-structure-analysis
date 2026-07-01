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
        return {"sigma_n": float("nan"), "mean_n": float("nan"), "n_success": 0, "samples": sample_arr}
    return {
        "sigma_n": float(np.nanstd(sample_arr, ddof=1)) if sample_arr.size > 1 else 0.0,
        "mean_n": float(np.nanmean(sample_arr)),
        "n_success": int(sample_arr.size),
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

    d_beta = (
        float(calculate_delta_n_rad(beta + steps["beta"], T, rho))
        - float(calculate_delta_n_rad(beta - steps["beta"], T, rho))
    ) / (2.0 * steps["beta"])
    d_T = (
        float(calculate_delta_n_rad(beta, T + steps["T"], rho))
        - float(calculate_delta_n_rad(beta, T - steps["T"], rho))
    ) / (2.0 * steps["T"])
    d_rho = (
        float(calculate_delta_n_rad(beta, T, rho + steps["rho"]))
        - float(calculate_delta_n_rad(beta, T, rho - steps["rho"]))
    ) / (2.0 * steps["rho"])

    sigma = np.sqrt((d_beta * sigma_beta) ** 2 + (d_T * sigma_T) ** 2 + (d_rho * sigma_rho) ** 2)
    return {
        "delta_n_rad": base,
        "sigma_delta_n_rad": float(sigma),
        "partial_beta": float(d_beta),
        "partial_T": float(d_T),
        "partial_rho": float(d_rho),
    }
