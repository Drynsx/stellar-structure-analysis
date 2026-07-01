"""CSV result persistence helpers retained for legacy workflows."""

from pathlib import Path

import pandas as pd


def _append(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def save_global_result(star_id, n_global, K, alpha, chi2, output_dir="outputs"):
    _append(Path(output_dir) / "results_global.csv", pd.DataFrame(
        [[star_id, n_global, K, alpha, chi2]],
        columns=["star_id", "n_global", "K", "alpha", "chi2"],
    ))


def save_local_result(star_id, r_array, n_local, output_dir="outputs"):
    _append(Path(output_dir) / "results_local.csv", pd.DataFrame(
        {"star_id": star_id, "normalized_radius": r_array, "n_local": n_local}
    ))


def save_piecewise_result(star_id, n_core, n_rad, n_conv, output_dir="outputs"):
    _append(Path(output_dir) / "results_piecewise.csv", pd.DataFrame(
        [[star_id, n_core, n_rad, n_conv]],
        columns=["star_id", "n_core", "n_radiative", "n_convective"],
    ))


def save_deviation_result(star_id, delta_rad, delta_mu, delta_conv, delta_nuc, delta_deg, delta_global, output_dir="outputs"):
    _append(Path(output_dir) / "results_deviations.csv", pd.DataFrame(
        [[star_id, delta_rad, delta_mu, delta_conv, delta_nuc, delta_deg, delta_global]],
        columns=["star_id", "delta_n_rad", "delta_n_mu", "delta_n_conv", "delta_n_nuc", "delta_n_deg", "delta_global"],
    ))


def save_profile(star_id, r_array, rho_array, pressure_array, temperature_array=None, output_dir="outputs/profiles"):
    data = {"normalized_radius": r_array, "density": rho_array, "pressure": pressure_array}
    if temperature_array is not None:
        data["temperature"] = temperature_array
    path = Path(output_dir) / f"star_{star_id}_profile.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)
