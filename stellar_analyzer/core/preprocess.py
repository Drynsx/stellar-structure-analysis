"""Preprocess stellar profiles into a validated 500-point CGS grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # SciPy is preferred, but the package can still run smoke checks without it.
    from scipy.interpolate import CubicSpline, interp1d
except Exception:  # pragma: no cover - exercised only in minimal/broken environments.
    CubicSpline = None
    interp1d = None

from stellar_analyzer.core.constants import G_CGS, M_SUN_CGS, R_SUN_CGS
from stellar_analyzer.core.data_loader import StellarModel


@dataclass
class PreprocessResult:
    profile: dict[str, np.ndarray]
    hydrostatic_ok: bool
    hydrostatic_max_relative_error: float
    issues: list[str] = field(default_factory=list)


def _as_profile_dict(model_or_profile: StellarModel | dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if isinstance(model_or_profile, StellarModel):
        return dict(model_or_profile.arrays), dict(model_or_profile.metadata)
    arrays = {key: np.asarray(value, dtype=float) for key, value in model_or_profile.items() if np.ndim(value) > 0}
    metadata = {key: value for key, value in model_or_profile.items() if np.ndim(value) == 0}
    return arrays, metadata


def _radius_scale(metadata: dict[str, Any]) -> float:
    for key in ("radius_cm", "stellar_radius_cm", "r_star_cm"):
        if key in metadata and float(metadata[key]) > 0:
            return float(metadata[key])
    for key in ("radius_rsun", "stellar_radius_rsun", "r_star_rsun"):
        if key in metadata and float(metadata[key]) > 0:
            return float(metadata[key]) * R_SUN_CGS
    return R_SUN_CGS


def _mass_scale(metadata: dict[str, Any]) -> float:
    for key in ("mass_g", "mass_cgs", "stellar_mass_g"):
        if key in metadata and float(metadata[key]) > 0:
            return float(metadata[key])
    for key in ("mass_msun", "stellar_mass_msun", "mass"):
        if key in metadata and float(metadata[key]) > 0 and float(metadata[key]) < 300:
            return float(metadata[key]) * M_SUN_CGS
    return M_SUN_CGS


def convert_to_cgs(profile: dict[str, np.ndarray], metadata: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Convert common profile fields to CGS units using conservative heuristics."""

    metadata = metadata or {}
    converted = {key: np.asarray(value, dtype=float).copy() for key, value in profile.items()}

    radius = converted.get("radius")
    radius_fraction = converted.get("radius_fraction")
    scale = _radius_scale(metadata)

    if radius is None and radius_fraction is not None:
        radius = radius_fraction * scale
    elif radius is not None:
        max_radius = float(np.nanmax(np.abs(radius)))
        if str(metadata.get("radius_unit", "")).lower() in {"rsun", "r_sun", "solar"}:
            radius = radius * R_SUN_CGS
        elif max_radius <= 2.5:
            radius_fraction = radius / max(max_radius, 1e-30)
            radius = radius_fraction * scale
        elif max_radius < 1e6:
            radius = radius * R_SUN_CGS
    else:
        raise ValueError("Profile must contain radius or radius_fraction")

    converted["radius"] = radius
    converted["radius_fraction"] = radius / max(float(np.nanmax(radius)), 1e-30)

    mass = converted.get("mass_enclosed")
    if mass is not None:
        max_mass = float(np.nanmax(np.abs(mass)))
        if str(metadata.get("mass_enclosed_unit", "")).lower() in {"msun", "m_sun", "solar"}:
            converted["mass_enclosed"] = mass * M_SUN_CGS
        elif max_mass <= 2.5:
            converted["mass_enclosed"] = mass * _mass_scale(metadata)
        elif max_mass < 1e10:
            converted["mass_enclosed"] = mass * M_SUN_CGS

    return converted


def _sort_and_deduplicate(profile: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    radius = np.asarray(profile["radius"], dtype=float)
    mask = np.isfinite(radius)
    for value in profile.values():
        if len(value) == len(radius):
            mask &= np.isfinite(value)

    order = np.argsort(radius[mask])
    unique_radius, unique_idx = np.unique(radius[mask][order], return_index=True)
    sorted_indices = np.flatnonzero(mask)[order][unique_idx]

    cleaned: dict[str, np.ndarray] = {}
    for key, value in profile.items():
        arr = np.asarray(value, dtype=float)
        if len(arr) == len(radius):
            cleaned[key] = arr[sorted_indices]
        else:
            cleaned[key] = arr
    cleaned["radius"] = unique_radius
    return cleaned


def _resample_array(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    if CubicSpline is not None and len(x_old) >= 4:
        try:
            return CubicSpline(x_old, y_old, extrapolate=True)(x_new)
        except Exception:
            pass
    if interp1d is not None:
        return interp1d(x_old, y_old, kind="linear", fill_value="extrapolate", bounds_error=False)(x_new)
    return np.interp(x_new, x_old, y_old)


def _integrate_mass(radius: np.ndarray, rho: np.ndarray) -> np.ndarray:
    shell_integrand = 4.0 * np.pi * radius**2 * rho
    mass = np.zeros_like(radius)
    dr = np.diff(radius)
    mass[1:] = np.cumsum(0.5 * (shell_integrand[1:] + shell_integrand[:-1]) * dr)
    return np.maximum.accumulate(mass)


def check_hydrostatic_equilibrium(profile: dict[str, np.ndarray]) -> tuple[bool, float]:
    """Check dP/dr ~= -G M(r) rho / r^2 using a scale-aware relative error."""

    radius = np.asarray(profile["radius"], dtype=float)
    pressure = np.asarray(profile["pressure"], dtype=float)
    rho = np.asarray(profile["rho"], dtype=float)
    mass = np.asarray(profile.get("mass_enclosed", _integrate_mass(radius, rho)), dtype=float)

    dP_dr = np.gradient(pressure, radius, edge_order=2)
    r_safe = np.maximum(radius, radius[1] if len(radius) > 1 else 1.0)
    rhs = -G_CGS * mass * rho / np.maximum(r_safe**2, 1e-30)

    scale = np.maximum.reduce([np.abs(dP_dr), np.abs(rhs), np.full_like(radius, 1e-99)])
    rel = np.abs(dP_dr - rhs) / scale
    finite_rel = rel[np.isfinite(rel)]
    max_error = float(np.nanmax(finite_rel)) if finite_rel.size else float("inf")
    return bool(max_error <= 0.05), max_error


def preprocess_profile(model_or_profile: StellarModel | dict[str, Any], n_points: int = 500) -> PreprocessResult:
    """Convert, validate, and resample a profile to exactly ``n_points`` samples."""

    issues: list[str] = []
    arrays, metadata = _as_profile_dict(model_or_profile)
    profile = convert_to_cgs(arrays, metadata)
    profile = _sort_and_deduplicate(profile)

    radius = profile["radius"]
    if len(radius) < 5:
        raise ValueError("At least five radial samples are required for preprocessing")
    if np.any(np.diff(radius) <= 0):
        raise ValueError("Radius grid must be strictly increasing after cleanup")

    for name in ("rho", "pressure", "temperature"):
        if name in profile and np.any(np.asarray(profile[name]) <= 0):
            raise ValueError(f"{name} must be strictly positive")

    radius_new = np.linspace(radius[0], radius[-1], n_points)
    resampled: dict[str, np.ndarray] = {"radius": radius_new}
    for key, value in profile.items():
        arr = np.asarray(value, dtype=float)
        if len(arr) == len(radius):
            resampled[key] = _resample_array(radius, arr, radius_new)

    resampled["radius_fraction"] = radius_new / max(radius_new[-1], 1e-30)

    for name in ("rho", "pressure", "temperature"):
        if name in resampled:
            floor = 1e-99 if name != "temperature" else 1.0
            resampled[name] = np.clip(resampled[name], floor, None)

    if "mass_enclosed" not in resampled and "rho" in resampled:
        resampled["mass_enclosed"] = _integrate_mass(resampled["radius"], resampled["rho"])

    hydrostatic_ok = False
    hydrostatic_error = float("inf")
    if {"rho", "pressure"}.issubset(resampled):
        hydrostatic_ok, hydrostatic_error = check_hydrostatic_equilibrium(resampled)
        if not hydrostatic_ok:
            issues.append("Hydrostatic-equilibrium residual exceeds 5 percent in at least one layer")

    return PreprocessResult(
        profile=resampled,
        hydrostatic_ok=hydrostatic_ok,
        hydrostatic_max_relative_error=hydrostatic_error,
        issues=issues,
    )
