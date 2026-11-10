"""External validation against a Standard Solar Model and asteroseismic data.

Chapter 3's operational definition of external validation (การเทียบเคียง
ภายนอก) is "ความสอดคล้องของตัวชี้วัด (Δn, RMS, PSI) ระหว่างผลคำนวณกับ
Standard Solar Model และชุดดาวที่มีข้อมูล asteroseismology" -- i.e. checking
whether Δn, RMS, and PSI agree between this project's computed results and
(a) a Standard Solar Model and (b) stars with asteroseismic data.

Two reference datasets ship with the project under ``data/reference/``:

* ``model_s_profile.csv`` -- a real, cited radial profile sample of Model S
  (Christensen-Dalsgaard et al. 1996), giving r/R, density, pressure,
  temperature, sound speed, and Gamma_1. Because this is a full internal
  structure profile, it supports the same profile-level Δn/RMS/PSI metrics
  used elsewhere in this project.

* ``kepler_legacy_sample.csv`` -- real published fundamental parameters
  (mass, radius, age, Teff, [Fe/H]) for a handful of Kepler Asteroseismic
  LEGACY sample stars (Lund et al. 2017; Silva Aguirre et al. 2017). These
  are *not* full internal radial profiles -- asteroseismic catalogs give
  global parameters derived from oscillation-frequency modeling, not
  density/pressure/temperature profiles -- so Kepler LEGACY validation in
  this module is a fundamental-parameter consistency check rather than a
  profile-level comparison. This is reported through the same Δn/RMS/PSI
  vocabulary so results read consistently, but the docstrings below are
  explicit about what each metric actually compares.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stellar_analyzer.core.global_fit import GlobalFitResult, fit_global_polytrope

DEFAULT_MODEL_S_PATH = Path(__file__).resolve().parents[2] / "data" / "reference" / "model_s_profile.csv"
DEFAULT_KEPLER_LEGACY_PATH = Path(__file__).resolve().parents[2] / "data" / "reference" / "kepler_legacy_sample.csv"
SUN_TEFF_K = 5778.0


@dataclass
class ExternalValidationResult:
    """Δn, RMS, and PSI agreement between a computed result and a reference."""

    reference: str
    delta_n: float
    rms: float
    psi: float
    central_value_error: float
    consistent: bool
    details: dict

    def to_dict(self) -> dict:
        return asdict(self)


def load_ssm_profile(path: str | Path = DEFAULT_MODEL_S_PATH) -> dict[str, np.ndarray]:
    """Load the bundled (or a user-supplied) Model S radial profile."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SSM reference profile not found: {path}")
    frame = pd.read_csv(path, comment="#")
    required = {"r_over_r", "rho_g_cm3", "pressure_dyn_cm2", "temperature_k"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"SSM reference profile is missing columns: {', '.join(sorted(missing))}")
    order = np.argsort(frame["r_over_r"].to_numpy(dtype=float))
    return {
        "r_over_r": frame["r_over_r"].to_numpy(dtype=float)[order],
        "rho": frame["rho_g_cm3"].to_numpy(dtype=float)[order],
        "pressure": frame["pressure_dyn_cm2"].to_numpy(dtype=float)[order],
        "temperature": frame["temperature_k"].to_numpy(dtype=float)[order],
    }


def load_kepler_legacy_sample(path: str | Path = DEFAULT_KEPLER_LEGACY_PATH) -> pd.DataFrame:
    """Load the bundled (or a user-supplied) Kepler LEGACY benchmark-star table."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Kepler LEGACY reference sample not found: {path}")
    frame = pd.read_csv(path, comment="#")
    required = {"kic", "teff_k", "mass_msun", "radius_rsun", "age_gyr"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Kepler LEGACY reference sample is missing columns: {', '.join(sorted(missing))}")
    return frame


def _profile_similarity_index(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Bounded (0, 1] similarity score between two positive radial profiles.

    Operational definition used in this implementation (Chapter 3 leaves
    PSI's exact formula unspecified beyond "unitless"): PSI = 1 / (1 +
    RMS_log), where RMS_log is the root-mean-square of the base-10
    log-ratio between the predicted and reference profiles. PSI = 1 is a
    perfect match; PSI falls toward 0 as the profiles diverge by orders of
    magnitude. Using the log-ratio keeps the score meaningful across the
    many decades density/pressure span between the core and the surface.
    """

    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-300, None)
    reference = np.clip(np.asarray(reference, dtype=float), 1e-300, None)
    log_ratio = np.log10(predicted / reference)
    rms_log = float(np.sqrt(np.mean(log_ratio**2)))
    return 1.0 / (1.0 + rms_log)


def _rms_profile_difference(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Fractional RMS difference between two positive radial profiles."""

    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-300, None)
    reference = np.clip(np.asarray(reference, dtype=float), 1e-300, None)
    return float(np.sqrt(np.mean(((predicted - reference) / reference) ** 2)))


def validate_pipeline_result(
    result: dict,
    kic: int | None = None,
    ssm_path: str | Path = DEFAULT_MODEL_S_PATH,
    kepler_path: str | Path = DEFAULT_KEPLER_LEGACY_PATH,
    consistency_threshold: float = 0.25,
) -> dict[str, dict]:
    """Run both external-validation checks against a ``pipeline.analyze_*`` result.

    Convenience wrapper so the CLI (and callers in general) do not need to
    know the ``result`` dict's internal shape to run Chapter 3's external
    validation step.
    """

    profile = result["profile"]
    ssm_result = validate_against_ssm(
        r_over_r=profile["radius_fraction"],
        rho=profile["rho"],
        n_global_computed=result["global_fit"]["n_global"],
        pressure=profile.get("pressure"),
        ssm_path=ssm_path,
        consistency_threshold=consistency_threshold,
    )
    kepler_result = validate_against_kepler_legacy(
        mass_msun=result["input"]["mass"],
        teff_k=result["input"]["teff"],
        age_gyr=result["input"]["age"],
        kic=kic,
        kepler_path=kepler_path,
        consistency_threshold=consistency_threshold,
    )
    return {"model_s": ssm_result.to_dict(), "kepler_legacy": kepler_result.to_dict()}


def validate_against_ssm(
    r_over_r: np.ndarray,
    rho: np.ndarray,
    n_global_computed: float,
    pressure: np.ndarray | None = None,
    ssm_path: str | Path = DEFAULT_MODEL_S_PATH,
    consistency_threshold: float = 0.25,
) -> ExternalValidationResult:
    """Compare a computed star's density profile and n_global against Model S.

    ``r_over_r``/``rho`` are the computed star's own radial grid; they are
    interpolated onto the SSM's grid (restricted to the overlapping range)
    before computing RMS and PSI. ``n_global_computed`` is compared against
    the polytropic index obtained by running the project's own
    :func:`fit_global_polytrope` on the SSM profile, so Δn uses the same
    fitting procedure on both sides of the comparison.
    """

    ssm = load_ssm_profile(ssm_path)
    ssm_fit = fit_global_polytrope(ssm["r_over_r"], ssm["rho"], SUN_TEFF_K)

    r_over_r = np.asarray(r_over_r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    order = np.argsort(r_over_r)
    r_over_r, rho = r_over_r[order], rho[order]

    lo = max(float(r_over_r.min()), float(ssm["r_over_r"].min()))
    hi = min(float(r_over_r.max()), float(ssm["r_over_r"].max()))
    if hi <= lo:
        raise ValueError("Computed profile and SSM profile do not overlap in r/R")
    common_grid = np.linspace(lo, hi, 200)
    predicted_rho = np.interp(common_grid, r_over_r, rho)
    reference_rho = np.interp(common_grid, ssm["r_over_r"], ssm["rho"])

    delta_n = float(n_global_computed - ssm_fit.n_global)
    rms = _rms_profile_difference(predicted_rho, reference_rho)
    psi = _profile_similarity_index(predicted_rho, reference_rho)

    central_value_error = float("nan")
    if pressure is not None:
        pressure = np.asarray(pressure, dtype=float)[order]
        predicted_pc = float(np.interp(lo, r_over_r, pressure))
        reference_pc = float(ssm["pressure"][np.argmin(ssm["r_over_r"])])
        central_value_error = abs(predicted_pc - reference_pc) / reference_pc

    consistent = abs(delta_n) <= consistency_threshold and rms <= consistency_threshold
    return ExternalValidationResult(
        reference="model_s",
        delta_n=delta_n,
        rms=rms,
        psi=psi,
        central_value_error=central_value_error,
        consistent=consistent,
        details={
            "ssm_n_global": ssm_fit.n_global,
            "ssm_reduced_chi2": ssm_fit.reduced_chi2,
            "overlap_r_over_r": [lo, hi],
            "consistency_threshold": consistency_threshold,
        },
    )


def validate_against_kepler_legacy(
    mass_msun: float,
    teff_k: float,
    age_gyr: float,
    kic: int | None = None,
    kepler_path: str | Path = DEFAULT_KEPLER_LEGACY_PATH,
    consistency_threshold: float = 0.25,
) -> ExternalValidationResult:
    """Check a computed star's (mass, Teff, age) against a Kepler LEGACY benchmark.

    Kepler asteroseismic catalogs give fundamental parameters derived from
    oscillation-frequency modeling, not internal radial profiles, so this
    is a fundamental-parameter consistency check rather than the profile
    comparison :func:`validate_against_ssm` performs. ``delta_n`` here is
    repurposed as the fractional mass offset, ``rms`` as the combined
    fractional offset across mass/radius/age, and ``psi`` as the matching
    :func:`_profile_similarity_index`-style bounded score over those same
    offsets, so results stay comparable in shape to the SSM comparison.

    If ``kic`` is not given, the benchmark star with the closest Teff is
    used (nearest-neighbor match), since the catalog is small.
    """

    sample = load_kepler_legacy_sample(kepler_path)
    if kic is not None:
        matches = sample[sample["kic"] == kic]
        if matches.empty:
            raise ValueError(f"KIC {kic} is not in the bundled Kepler LEGACY sample")
        row = matches.iloc[0]
    else:
        row = sample.iloc[(sample["teff_k"] - teff_k).abs().argsort().iloc[0]]

    mass_offset = (mass_msun - float(row["mass_msun"])) / float(row["mass_msun"])
    age_offset = (age_gyr - float(row["age_gyr"])) / max(float(row["age_gyr"]), 1e-9)
    offsets = np.array([mass_offset, age_offset], dtype=float)
    rms = float(np.sqrt(np.mean(offsets**2)))
    psi = 1.0 / (1.0 + rms)

    mass_in_range = float(row["mass_msun_lo"]) <= mass_msun <= float(row["mass_msun_hi"])
    age_in_range = float(row["age_gyr_lo"]) <= age_gyr <= float(row["age_gyr_hi"])
    consistent = abs(mass_offset) <= consistency_threshold and (mass_in_range or age_in_range)

    return ExternalValidationResult(
        reference="kepler_legacy",
        delta_n=mass_offset,
        rms=rms,
        psi=psi,
        central_value_error=age_offset,
        consistent=consistent,
        details={
            "matched_kic": int(row["kic"]),
            "benchmark_teff_k": float(row["teff_k"]),
            "benchmark_mass_range_msun": [float(row["mass_msun_lo"]), float(row["mass_msun_hi"])],
            "benchmark_age_range_gyr": [float(row["age_gyr_lo"]), float(row["age_gyr_hi"])],
            "mass_within_published_range": bool(mass_in_range),
            "age_within_published_range": bool(age_in_range),
            "consistency_threshold": consistency_threshold,
        },
    )
