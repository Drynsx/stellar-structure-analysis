"""Build the reproducible evidence package used by the research report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stellar_analyzer.core.data_loader import list_mesa_profiles, load_mesa_web_job
from stellar_analyzer.core.pipeline import analyze_profile
from stellar_analyzer.core.uncertainty import propagate_delta_n_rad_error


DEFAULT_JOB = PROJECT_ROOT / "data" / "raw" / "MESA-Web_Job_03242664908"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_report_data"
PAPER_REFERENCE = Path(r"C:\Users\Lenovo\Documents\physics sheets\บท1-3 (fixed format ครั้ง 1).pdf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _clean_output(output: Path) -> None:
    if output.exists():
        resolved = output.resolve()
        if resolved.parent != PROJECT_ROOT.resolve() or resolved.name != "paper_report_data":
            raise ValueError(f"Refusing to replace unexpected output directory: {resolved}")
        shutil.rmtree(resolved)
    output.mkdir(parents=True)


def _reference_record() -> dict:
    record = {
        "title_en": "Development of a Computational System for Stellar Structure Analysis and Anomaly Detection Using Polytropic Indices and Machine Learning",
        "reference_file": PAPER_REFERENCE.name,
        "reference_available": PAPER_REFERENCE.exists(),
        "reference_pages": 43,
        "method_pages": "37-42",
        "objectives_pages": "3-4",
    }
    if PAPER_REFERENCE.exists():
        record.update({"sha256": _sha256(PAPER_REFERENCE), "size_bytes": PAPER_REFERENCE.stat().st_size})
    return record


def build(job: Path, output: Path) -> None:
    _clean_output(output)
    (output / "README.md").write_text(
        """# Paper Report Data

This directory is generated from the legacy MESA-Web job by:

```powershell
.venv\\Scripts\\python.exe scripts\\build_paper_report_data.py
```

The numbered folders follow the ten computational steps on pages 38-42 of the
provided research-paper reference. `METHOD_TRACEABILITY.csv` maps each method to
its evidence, and `MANIFEST.csv` records file sizes and SHA-256 checksums.

Important interpretation notes:

- These eight snapshots are from one approximately solar-mass MESA track. They
  are implementation evidence, not the proposed 50,000-model atlas.
- The PINN is not trained; no synthetic training result is presented as evidence.
- Hydrostatic failures are retained as findings rather than silently removed.
- Radiation uncertainty uses a recorded 1% input-error assumption. The
  1,000-resample bootstrap protocol is documented but not claimed as completed.
""",
        encoding="utf-8",
    )
    snapshots = list_mesa_profiles(job)
    if not snapshots:
        raise ValueError(f"No MESA profiles found in {job}")

    _write_json(output / "00_reference" / "paper_reference.json", _reference_record())

    source_rows: list[dict] = []
    global_rows: list[dict] = []
    quality_rows: list[dict] = []
    global_fit_rows: list[dict] = []
    piecewise_rows: list[dict] = []
    local_rows: list[dict] = []
    deviation_rows: list[dict] = []
    contribution_rows: list[dict] = []
    residual_rows: list[dict] = []
    uncertainty_rows: list[dict] = []

    for snapshot in snapshots:
        profile_number = int(snapshot["profile_number"])
        model = load_mesa_web_job(job, profile_number)
        result = analyze_profile(model)
        metadata = model.metadata
        profile = {key: np.asarray(value, dtype=float) for key, value in result["profile"].items()}
        star_id = f"mesa_model_{int(snapshot['model_number']):04d}_profile_{profile_number:02d}"

        source_path = Path(snapshot["path"])
        source_rows.append(
            {
                "star_id": star_id,
                "profile_number": profile_number,
                "model_number": int(snapshot["model_number"]),
                "priority": int(snapshot["priority"]),
                "source_file": _display_path(source_path),
                "size_bytes": source_path.stat().st_size,
                "sha256": _sha256(source_path),
            }
        )
        global_rows.append(
            {
                "star_id": star_id,
                "model_number": int(metadata["model_number"]),
                "mass_msun": float(metadata["star_mass"]),
                "radius_rsun": float(metadata["photosphere_r"]),
                "teff_k": float(metadata["teff"]),
                "luminosity_lsun": float(metadata["photosphere_l"]),
                "age_years": float(metadata["star_age"]),
                "initial_z": float(metadata["initial_z"]),
                "evolution_stage": "pre-main-sequence" if float(metadata["star_age"]) < 1e8 else "main-sequence",
                "n_points_raw": int(metadata["num_zones"]),
                "n_points_processed": len(profile["radius"]),
            }
        )

        profile_frame = pd.DataFrame(
            {
                "star_id": star_id,
                "radius_fraction": profile["radius_fraction"],
                "radius_cm": profile["radius"],
                "mass_enclosed_g": profile["mass_enclosed"],
                "density_g_cm3": profile["rho"],
                "pressure_dyn_cm2": profile["pressure"],
                "temperature_k": profile["temperature"],
                "mean_molecular_weight": profile["mu"],
                "epsilon_erg_g_s": profile["epsilon"],
                "grad_ad": profile["grad_ad"],
                "grad_rad": profile["grad_rad"],
            }
        )
        profile_path = output / "02_preprocessing" / "profiles" / f"{star_id}.csv"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_frame.to_csv(profile_path, index=False)

        quality_rows.append(
            {
                "star_id": star_id,
                "radius_monotonic": bool(np.all(np.diff(profile["radius"]) > 0)),
                "min_dr_cm": float(np.min(np.diff(profile["radius"]))),
                "density_positive": bool(np.all(profile["rho"] > 0)),
                "pressure_positive": bool(np.all(profile["pressure"] > 0)),
                "temperature_positive": bool(np.all(profile["temperature"] > 0)),
                "hydrostatic_ok_5_percent": bool(result["preprocessing"]["hydrostatic_ok"]),
                "hydrostatic_max_relative_error": float(result["preprocessing"]["hydrostatic_max_relative_error"]),
                "issues": " | ".join(result["preprocessing"]["issues"]),
            }
        )

        global_fit_rows.append({"star_id": star_id, **result["global_fit"]})
        piecewise_rows.append({"star_id": star_id, **result["piecewise_fit"]})
        local_rows.extend(
            {
                "star_id": star_id,
                "radius_fraction": radius_fraction,
                "n_local": n_local,
            }
            for radius_fraction, n_local in zip(profile["radius_fraction"], result["n_local"])
        )

        factors = result["deviation_factors"]
        deviation_rows.append({"star_id": star_id, **factors})
        denominator = sum(abs(float(value)) for value in factors.values())
        contribution_rows.extend(
            {
                "star_id": star_id,
                "driver": name,
                "delta_n": float(value),
                "absolute_contribution_percent": 100.0 * abs(float(value)) / denominator if denominator else 0.0,
            }
            for name, value in factors.items()
        )
        residual_rows.append(
            {
                "star_id": star_id,
                "n_global_observed": result["global_fit"]["n_global"],
                "n_base": result["global_fit"]["n_global"] - result["residual"]["delta_n_obs"],
                "delta_n_observed": result["residual"]["delta_n_obs"],
                "delta_n_theory": result["residual"]["delta_n_theory"],
                "delta_global": result["residual"]["delta_global"],
                "status": result["status"],
            }
        )

        center = 0
        rho = float(profile["rho"][center])
        temperature = float(profile["temperature"][center])
        gas_pressure = rho * 1.380649e-16 * temperature / (float(profile["mu"][center]) * 1.67262192369e-24)
        beta = float(np.clip(gas_pressure / profile["pressure"][center], 1e-6, 1.0))
        propagated = propagate_delta_n_rad_error(
            beta,
            temperature,
            rho,
            sigma_beta=0.01 * beta,
            sigma_T=0.01 * temperature,
            sigma_rho=0.01 * rho,
        )
        uncertainty_rows.append(
            {
                "star_id": star_id,
                "sample_location": "innermost_processed_layer",
                "assumed_relative_input_uncertainty": 0.01,
                "beta": beta,
                "temperature_k": temperature,
                "density_g_cm3": rho,
                **propagated,
            }
        )

    _write_csv(output / "01_input" / "source_manifest.csv", source_rows)
    _write_csv(output / "01_input" / "global_parameters.csv", global_rows)
    _write_csv(output / "02_preprocessing" / "quality_checks.csv", quality_rows)
    _write_csv(output / "03_polytropic_indices" / "global_fits.csv", global_fit_rows)
    _write_csv(output / "03_polytropic_indices" / "piecewise_fits.csv", piecewise_rows)
    _write_csv(output / "03_polytropic_indices" / "local_indices.csv", local_rows)
    _write_csv(output / "04_deviation_drivers" / "deviation_summary.csv", deviation_rows)
    _write_csv(output / "04_deviation_drivers" / "contribution_percentages.csv", contribution_rows)
    _write_csv(output / "05_global_comparison" / "global_residuals.csv", residual_rows)
    _write_csv(output / "06_uncertainty" / "radiation_error_propagation.csv", uncertainty_rows)

    _write_json(
        output / "06_uncertainty" / "bootstrap_protocol.json",
        {
            "method": "resample radial-density pairs with replacement and refit n_global",
            "required_resamples": 1000,
            "implemented_function": "stellar_analyzer.core.uncertainty.bootstrap_global_n",
            "status": "not_run_for_this_evidence_build",
            "reason": "Bootstrap is intentionally kept separate because it is stochastic and computationally expensive.",
            "reproduction_command": ".venv\\Scripts\\python.exe -m stellar_analyzer.cli uncertainty --bootstrap 1000 (CLI pending)",
        },
    )
    _write_json(
        output / "07_machine_learning" / "training_readiness.json",
        {
            "pinn_trained": False,
            "available_models": len(snapshots),
            "available_stellar_tracks": 1,
            "paper_target_models": 50000,
            "architecture_implemented": True,
            "split_required": {"training_percent": 70, "validation_percent": 15, "test_percent": 15},
            "training_required": {"epochs": 300, "batch_size": 32, "learning_rate": 0.001, "patience": 20},
            "blocking_reason": "One solar-mass MESA track is not sufficient for a generalizable PINN.",
        },
    )
    _write_json(
        output / "08_storage" / "storage_evidence.json",
        {
            "database_models": ["stars", "profiles", "results"],
            "implementation": "stellar_analyzer.web.database",
            "local_array_format": "CSV evidence package; HDF5 supported by loader",
            "production_database": "PostgreSQL via SQLAlchemy",
            "database_deployed": False,
        },
    )
    _write_json(
        output / "09_validation" / "validation_summary.json",
        {
            "profiles_processed": len(snapshots),
            "all_profiles_have_500_points": all(row["n_points_processed"] == 500 for row in global_rows),
            "all_radii_monotonic": all(row["radius_monotonic"] for row in quality_rows),
            "all_scalar_profiles_positive": all(
                row["density_positive"] and row["pressure_positive"] and row["temperature_positive"]
                for row in quality_rows
            ),
            "hydrostatic_pass_count": sum(row["hydrostatic_ok_5_percent"] for row in quality_rows),
            "hydrostatic_fail_count": sum(not row["hydrostatic_ok_5_percent"] for row in quality_rows),
            "test_command": ".venv\\Scripts\\python.exe -m pytest -q",
            "python_version": platform.python_version(),
        },
    )
    _write_json(
        output / "10_monitoring" / "reproducibility.json",
        {
            "builder": "scripts/build_paper_report_data.py",
            "source_job": _display_path(job),
            "deterministic_steps": "Steps 1-5 and 8-10",
            "stochastic_step": "Step 7 bootstrap; random seed must be recorded when run",
            "generated_file_count": sum(1 for path in output.rglob("*") if path.is_file()),
        },
    )

    traceability = [
        {"paper_step": 1, "paper_pages": "38", "method": "Data input handling", "evidence": "01_input/global_parameters.csv; 01_input/source_manifest.csv"},
        {"paper_step": 2, "paper_pages": "38-39", "method": "Read source and convert units", "evidence": "01_input/source_manifest.csv; 02_preprocessing/profiles/"},
        {"paper_step": 3, "paper_pages": "39", "method": "Validate and resample", "evidence": "02_preprocessing/quality_checks.csv"},
        {"paper_step": 4, "paper_pages": "39-40", "method": "Global, local, and zonal indices", "evidence": "03_polytropic_indices/"},
        {"paper_step": 5, "paper_pages": "40", "method": "Five deviation drivers", "evidence": "04_deviation_drivers/"},
        {"paper_step": 6, "paper_pages": "40-41", "method": "PINN training", "evidence": "07_machine_learning/training_readiness.json"},
        {"paper_step": 7, "paper_pages": "41", "method": "Uncertainty", "evidence": "06_uncertainty/"},
        {"paper_step": 8, "paper_pages": "41-42", "method": "Database and storage", "evidence": "08_storage/storage_evidence.json"},
        {"paper_step": 9, "paper_pages": "42", "method": "Testing and validation", "evidence": "09_validation/validation_summary.json"},
        {"paper_step": 10, "paper_pages": "42", "method": "Logging and monitoring", "evidence": "10_monitoring/reproducibility.json"},
    ]
    _write_csv(output / "METHOD_TRACEABILITY.csv", traceability)

    manifest_rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        manifest_rows.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_csv(output / "MANIFEST.csv", manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.job, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
