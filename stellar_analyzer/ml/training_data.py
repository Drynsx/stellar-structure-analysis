"""Reproducible dataset preparation for stellar PINN training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stellar_analyzer.core.data_loader import StellarModel, iter_hdf5_profiles, list_mesa_profiles, load_mesa_web_job
from stellar_analyzer.core.pipeline import analyze_profile
from stellar_analyzer.ml.pinn_model import DELTA_NAMES, build_input_features, transform_delta_n


TARGET_SCHEMA = (
    "theta=(rho/rho_center)^(1/n_global)",
    "log10(rho/rho_center)/12",
    "log10(pressure/pressure_center)/18",
)


def _portable_source_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.name


def build_model_tensors(model: StellarModel, n_points: int):
    result = analyze_profile(model, n_points=n_points)
    profile = result["profile"]
    inputs = result["input"]
    radius = np.asarray(profile["radius_fraction"], dtype=np.float32)
    shape = np.ones_like(radius)
    feature = build_input_features(
        inputs["mass"] * shape, inputs["teff"] * shape, inputs["metallicity"] * shape,
        inputs["age"] * shape, radius,
    ).astype(np.float32)
    rho = np.asarray(profile["rho"], dtype=np.float64)
    pressure = np.asarray(profile["pressure"], dtype=np.float64)
    n_global = max(float(result["global_fit"]["n_global"]), 0.1)
    target_profile = np.column_stack([
        np.power(np.clip(rho / rho[0], 0.0, 1.0), 1.0 / n_global),
        np.log10(np.clip(rho / rho[0], 1e-12, None)) / 12.0,
        np.log10(np.clip(pressure / pressure[0], 1e-18, None)) / 18.0,
    ]).astype(np.float32)
    raw_delta = np.asarray([result["deviation_factors"][key] for key in DELTA_NAMES], dtype=np.float32)
    return feature, target_profile, transform_delta_n(raw_delta).astype(np.float32), raw_delta, n_global


def prepare_mesa_dataset(job_path: str | Path, output_path: str | Path, n_points: int = 500) -> dict:
    """Convert every snapshot in a MESA-Web job into sequence tensors."""

    job = Path(job_path)
    snapshots = list_mesa_profiles(job)
    if len(snapshots) < 3:
        raise ValueError("At least three independent snapshots are required for train/validation/test splits")

    features, profile_targets, delta_targets, n_indices = [], [], [], []
    profile_numbers, model_numbers = [], []
    for snapshot in snapshots:
        model = load_mesa_web_job(job, snapshot["profile_number"])
        feature, target_profile, delta, raw_delta, n_global = build_model_tensors(model, n_points)

        features.append(feature)
        profile_targets.append(target_profile)
        delta_targets.append(raw_delta)
        n_indices.append(n_global)
        profile_numbers.append(snapshot["profile_number"])
        model_numbers.append(snapshot["model_number"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_job": _portable_source_path(job),
        "samples": len(snapshots),
        "radial_points": n_points,
        "feature_count": 15,
        "target_schema": TARGET_SCHEMA,
        "delta_names": DELTA_NAMES,
        "delta_transform": "sign(x) * log1p(abs(x)); inverse: sign(y) * expm1(abs(y))",
        "generalization_warning": "One stellar track is suitable for pipeline validation, not a production model.",
    }
    np.savez_compressed(
        output,
        features=np.stack(features),
        profiles=np.stack(profile_targets),
        delta_n=transform_delta_n(np.asarray(delta_targets, dtype=np.float32)).astype(np.float32),
        delta_n_raw=np.asarray(delta_targets, dtype=np.float32),
        n_index=np.asarray(n_indices, dtype=np.float32),
        profile_number=np.asarray(profile_numbers, dtype=np.int32),
        model_number=np.asarray(model_numbers, dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata)),
    )
    return {**metadata, "output": str(output), "shape": list(np.stack(features).shape)}


def prepare_hdf5_grid_dataset(
    grid_path: str | Path,
    output_path: str | Path,
    n_points: int = 500,
    limit: int | None = None,
) -> dict:
    """Stream a radial-profile HDF5 grid into a chunked training dataset."""

    import h5py

    source = Path(grid_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    metadata = {
        "source_grid": _portable_source_path(source), "radial_points": n_points, "feature_count": 15,
        "target_schema": TARGET_SCHEMA, "delta_names": DELTA_NAMES,
        "delta_transform": "sign(x) * log1p(abs(x)); inverse: sign(y) * expm1(abs(y))",
    }
    with h5py.File(output, "w") as target:
        features = target.create_dataset("features", (0, n_points, 15), maxshape=(None, n_points, 15),
                                         chunks=(1, n_points, 15), dtype="f4", compression="gzip")
        profiles = target.create_dataset("profiles", (0, n_points, 3), maxshape=(None, n_points, 3),
                                         chunks=(1, n_points, 3), dtype="f4", compression="gzip")
        delta_n = target.create_dataset("delta_n", (0, 5), maxshape=(None, 5), chunks=(256, 5), dtype="f4")
        delta_raw = target.create_dataset("delta_n_raw", (0, 5), maxshape=(None, 5), chunks=(256, 5), dtype="f4")
        n_index = target.create_dataset("n_index", (0,), maxshape=(None,), chunks=(256,), dtype="f4")
        for model in iter_hdf5_profiles(source, limit=limit):
            feature, profile, delta, raw_delta, fitted_n = build_model_tensors(model, n_points)
            for dataset in (features, profiles, delta_n, delta_raw, n_index):
                dataset.resize(count + 1, axis=0)
            features[count], profiles[count] = feature, profile
            delta_n[count], delta_raw[count], n_index[count] = delta, raw_delta, fitted_n
            count += 1
        if count < 3:
            raise ValueError("At least three radial profiles are required for train/validation/test splits")
        metadata["samples"] = count
        target.attrs["metadata"] = json.dumps(metadata)
    return {**metadata, "output": str(output), "shape": [count, n_points, 15]}


def inspect_dataset(path: str | Path) -> dict:
    """Return safe metadata and shapes without loading pickled objects."""

    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5", ".hdf"}:
        import h5py
        with h5py.File(path, "r") as data:
            metadata = json.loads(data.attrs["metadata"])
            return {**metadata, "features_shape": list(data["features"].shape),
                    "profiles_shape": list(data["profiles"].shape), "delta_shape": list(data["delta_n"].shape)}
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        return {
            **metadata,
            "features_shape": list(data["features"].shape),
            "profiles_shape": list(data["profiles"].shape),
            "delta_shape": list(data["delta_n"].shape),
        }
