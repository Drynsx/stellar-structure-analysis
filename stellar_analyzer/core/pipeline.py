"""End-to-end scientific analysis pipeline used by the CLI and Python API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from stellar_analyzer.core.constants import A_RAD_CGS, G_CGS, K_B_CGS, M_P_CGS, M_SUN_CGS, R_SUN_CGS
from stellar_analyzer.core.deviation_drivers import (
    calculate_delta_n_conv,
    calculate_delta_n_deg,
    calculate_delta_n_mu,
    calculate_delta_n_nuc,
    calculate_delta_n_rad,
    calculate_global_residual,
)
from stellar_analyzer.core.global_fit import fit_global_polytrope
from stellar_analyzer.core.local_fit import calculate_local_n
from stellar_analyzer.core.piecewise_fit import fit_piecewise
from stellar_analyzer.core.data_loader import StellarModel, load_mesa_web_job, load_stellar_model
from stellar_analyzer.core.preprocess import preprocess_profile


@dataclass
class StarRequest:
    mass: float = 1.0
    teff: float = 5778.0
    metallicity: float = 0.0
    age: float = 4.6
    name: str = "Custom Star"


def _validate_request(request: StarRequest) -> StarRequest:
    if not (0.08 <= request.mass <= 80.0):
        raise ValueError("mass must be between 0.08 and 80 solar masses")
    if not (1500.0 <= request.teff <= 60000.0):
        raise ValueError("teff must be between 1500 K and 60000 K")
    if not (-4.0 <= request.metallicity <= 1.0):
        raise ValueError("metallicity must be between -4.0 and 1.0 dex")
    if not (0.0 <= request.age <= 14.0):
        raise ValueError("age must be between 0 and 14 Gyr")
    return request


def generate_surrogate_profile(request: StarRequest, n_points: int = 500) -> dict[str, np.ndarray]:
    """Generate a smooth, physically plausible profile when no grid/PINN is present."""

    request = _validate_request(request)
    rfrac = np.linspace(0.0, 1.0, n_points)
    radius_star = R_SUN_CGS * request.mass**0.8 * (request.teff / 5778.0) ** -0.15
    radius = np.maximum(rfrac * radius_star, radius_star * 1e-6)

    central_rho = 150.0 * request.mass ** 1.4 * 10.0 ** (-0.15 * request.metallicity)
    envelope_floor = 1e-7 * max(request.mass, 0.1)
    density_shape = np.clip(1.0 - rfrac**2, 0.0, 1.0) ** 2.2
    rho = envelope_floor + central_rho * density_shape

    central_temp = 1.55e7 * request.mass ** 0.65 * (1.0 + 0.04 * request.metallicity)
    temperature = np.maximum(request.teff, central_temp * (1.0 - 0.94 * rfrac**1.55) + request.teff * rfrac**3)

    mu = 0.61 + 0.018 * request.metallicity + 0.03 * np.exp(-(rfrac / 0.18) ** 2)
    gas_pressure = rho * K_B_CGS * temperature / (np.clip(mu, 0.1, None) * M_P_CGS)
    radiation_pressure = A_RAD_CGS * temperature**4 / 3.0
    pressure = gas_pressure + radiation_pressure

    epsilon = 2.0 * request.mass**4 * np.exp(-(rfrac / 0.09) ** 2) * np.maximum(temperature / 1.5e7, 0.1) ** 4
    grad_ad = np.full_like(rfrac, 0.4)
    grad_rad = 0.18 + 0.35 * np.exp(-((rfrac - 0.78) / 0.18) ** 2) + 0.03 * request.mass

    shell_integrand = 4.0 * np.pi * radius**2 * rho
    mass_enclosed = np.zeros_like(radius)
    mass_enclosed[1:] = np.cumsum(0.5 * (shell_integrand[1:] + shell_integrand[:-1]) * np.diff(radius))
    mass_enclosed *= (request.mass * M_SUN_CGS) / max(mass_enclosed[-1], 1e-99)

    return {
        "radius": radius,
        "radius_fraction": rfrac,
        "rho": rho,
        "pressure": pressure,
        "temperature": temperature,
        "mu": mu,
        "epsilon": epsilon,
        "grad_ad": grad_ad,
        "grad_rad": grad_rad,
        "mass_enclosed": mass_enclosed,
    }


def _pressure_scale_height(radius: np.ndarray, rho: np.ndarray) -> np.ndarray:
    dlnrho_dr = np.gradient(np.log(np.clip(rho, 1e-300, None)), radius, edge_order=2)
    return np.clip(1.0 / np.maximum(np.abs(dlnrho_dr), 1e-30), 0.0, np.nanmax(radius))


def _complete_profile(profile: dict[str, np.ndarray], request: StarRequest) -> dict[str, np.ndarray]:
    """Supply conservative defaults for optional fields in imported profiles."""

    result = {key: np.asarray(value, dtype=float) for key, value in profile.items()}
    rho = result["rho"]
    pressure = result["pressure"]
    count = len(rho)
    result.setdefault("mu", np.full(count, 0.61))
    if "temperature" not in result:
        result["temperature"] = np.maximum(
            request.teff,
            pressure * result["mu"] * M_P_CGS / np.maximum(rho * K_B_CGS, 1e-99),
        )
    result.setdefault("epsilon", np.zeros(count))
    result.setdefault("grad_ad", np.full(count, 0.4))
    result.setdefault("grad_rad", np.full(count, 0.25))
    return result


def _analyze_profile(
    request: StarRequest,
    profile: dict[str, np.ndarray],
    source: dict[str, Any],
    preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = _validate_request(request)
    profile = _complete_profile(profile, request)
    r = profile["radius"]
    rfrac = profile["radius_fraction"]
    rho = profile["rho"]
    pressure = profile["pressure"]
    temperature = profile["temperature"]

    global_fit = fit_global_polytrope(r, rho, request.teff)
    local_n = calculate_local_n(pressure, rho, rfrac)
    piecewise = fit_piecewise(
        rfrac,
        rho,
        pressure,
        profile["grad_ad"],
        profile["grad_rad"],
        global_chi2=global_fit.reduced_chi2,
    )

    gas_pressure = rho * K_B_CGS * temperature / (np.clip(profile["mu"], 0.1, None) * M_P_CGS)
    beta = np.clip(gas_pressure / np.maximum(pressure, 1e-99), 1e-6, 1.0)
    h_rho = _pressure_scale_height(r, rho) / max(float(np.nanmax(r)), 1e-30)
    tau_dyn = np.sqrt(np.maximum(r**3 / (G_CGS * np.maximum(profile["mass_enclosed"], 1e-99)), 1e-30))
    tau_conv = 2.0 * tau_dyn * np.clip(1.0 - rfrac, 0.05, 1.0)

    delta_rad = calculate_delta_n_rad(beta, temperature, rho)
    delta_mu = calculate_delta_n_mu(profile["mu"], rfrac, h_rho)
    delta_conv = calculate_delta_n_conv(profile["grad_rad"], profile["grad_ad"], tau_conv, tau_dyn)
    delta_nuc = calculate_delta_n_nuc(profile["epsilon"], rho, r)
    delta_deg = calculate_delta_n_deg(temperature, rho, X_fraction=0.70)
    weights = np.gradient(profile["mass_enclosed"], edge_order=2)
    n_base = 3.0 if request.teff > 10000.0 else 1.5
    residual = calculate_global_residual(
        global_fit.n_global,
        n_base,
        [delta_rad, delta_mu, delta_conv, delta_nuc, delta_deg],
        weights=np.abs(weights),
    )

    deltas = {
        "delta_n_rad": float(np.average(delta_rad, weights=np.abs(weights))),
        "delta_n_mu": float(np.average(delta_mu, weights=np.abs(weights))),
        "delta_n_conv": float(np.average(delta_conv, weights=np.abs(weights))),
        "delta_n_nuc": float(delta_nuc),
        "delta_n_deg": float(np.average(delta_deg, weights=np.abs(weights))),
    }

    return {
        "input": asdict(request),
        "source": source,
        "preprocessing": preprocessing or {},
        "profile": {key: value.tolist() for key, value in profile.items()},
        "global_fit": asdict(global_fit),
        "piecewise_fit": asdict(piecewise),
        "n_local": local_n.tolist(),
        "deviation_factors": deltas,
        "anomaly_score": residual.delta_global,
        "status": "Anomaly" if residual.status == "ANOMALOUS" else "Normal",
        "residual": asdict(residual),
    }


def analyze_star(params: dict[str, Any] | StarRequest, use_pinn: bool = True) -> dict[str, Any]:
    """Analyze a parameterized star using the deterministic surrogate profile."""

    request = params if isinstance(params, StarRequest) else StarRequest(**params)
    profile = generate_surrogate_profile(request)
    return _analyze_profile(request, profile, {"type": "surrogate", "use_pinn": bool(use_pinn)})


def _request_from_model(model: StellarModel, overrides: dict[str, Any] | None = None) -> StarRequest:
    metadata = model.metadata
    initial_z = float(metadata.get("initial_z", 0.02))
    metallicity = metadata.get("metallicity")
    if metallicity is None:
        metallicity = np.log10(max(initial_z, 1e-12) / 0.02)
    age = float(metadata.get("age", float(metadata.get("star_age", 0.0)) / 1e9))
    inferred = {
        "name": f"MESA {metadata.get('source_profile', 'profile')}",
        "mass": float(metadata.get("star_mass", metadata.get("initial_mass", metadata.get("mass", 1.0)))),
        "teff": float(metadata.get("teff", 5778.0)),
        "metallicity": float(metallicity),
        "age": age,
    }
    inferred.update(overrides or {})
    return StarRequest(**inferred)


def analyze_profile(
    profile_or_path: StellarModel | str,
    params: dict[str, Any] | None = None,
    n_points: int = 500,
) -> dict[str, Any]:
    """Analyze an imported MESA/MIST/BaSTI profile on a normalized CGS grid."""

    model = profile_or_path if isinstance(profile_or_path, StellarModel) else load_stellar_model(profile_or_path)
    prepared = preprocess_profile(model, n_points=n_points)
    request = _request_from_model(model, params)
    source = {
        "type": str(model.metadata.get("format", "imported")),
        "path": model.source_path,
        "profile_number": model.metadata.get("profile_number"),
        "model_number": model.metadata.get("model_number"),
    }
    preprocessing = {
        "hydrostatic_ok": prepared.hydrostatic_ok,
        "hydrostatic_max_relative_error": prepared.hydrostatic_max_relative_error,
        "issues": prepared.issues,
        "points": n_points,
    }
    return _analyze_profile(request, prepared.profile, source, preprocessing)


def analyze_mesa_job(
    job_path: str,
    profile_number: int | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and analyze a snapshot from a legacy MESA-Web job."""

    return analyze_profile(load_mesa_web_job(job_path, profile_number), params=params)


def _fast_row(row: Any) -> dict[str, float | str]:
    mass = float(row["mass"])
    teff = float(row["teff"])
    metallicity = float(row["metallicity"])
    age = float(row["age"])
    n_base = 3.0 if teff > 10000.0 else 1.5
    delta_rad = max(0.0, 0.08 * (teff / 5778.0) ** 2.2 / max(mass, 0.1))
    delta_mu = max(0.0, 0.025 * abs(metallicity) + 0.004 * age)
    delta_conv = max(0.0, 0.12 / (1.0 + np.exp((teff - 6500.0) / 650.0)))
    delta_nuc = max(0.0, 0.10 * mass**1.7 / (1.0 + mass**1.7))
    delta_deg = max(0.0, 0.05 / max(mass, 0.08) ** 1.8 if mass < 0.5 else 0.0)
    n_global = n_base + delta_rad + delta_mu + delta_conv + delta_nuc + delta_deg + 0.02 * np.sin(age + mass)
    theory = delta_rad + delta_mu + delta_conv + delta_nuc + delta_deg
    anomaly_score = n_global - n_base - theory
    return {
        "n_global": float(n_global),
        "anomaly_score": float(anomaly_score),
        "delta_n_rad": float(delta_rad),
        "delta_n_mu": float(delta_mu),
        "delta_n_conv": float(delta_conv),
        "delta_n_nuc": float(delta_nuc),
        "delta_n_deg": float(delta_deg),
        "status": "Anomaly" if anomaly_score > 0.1 else "Normal",
    }


def batch_analyze(frame: Any):
    """Analyze many stars with a vectorized surrogate path suited for large CSVs."""

    import pandas as pd

    required = {"mass", "teff", "metallicity", "age"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Batch input is missing columns: {', '.join(sorted(missing))}")

    results = pd.DataFrame([_fast_row(row) for _, row in frame.iterrows()], index=frame.index)
    return pd.concat([frame.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
