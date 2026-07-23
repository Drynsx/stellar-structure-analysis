"""Build the reproducible evidence package used by the research report."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
from stellar_analyzer.core.constants import G_CGS, K_B_CGS, M_P_CGS
from stellar_analyzer.core.deviation_drivers import (
    ANOMALY_THRESHOLD,
    calculate_delta_n_conv,
    calculate_delta_n_deg,
    calculate_delta_n_mu,
    calculate_delta_n_nuc,
    calculate_delta_n_rad,
)
from stellar_analyzer.core.pipeline import analyze_profile
from stellar_analyzer.core.uncertainty import bootstrap_global_n, propagate_delta_n_rad_error
from stellar_analyzer.core.validation import assess_multitrack_manifest, leave_one_track_out


DEFAULT_JOB = PROJECT_ROOT / "data" / "raw" / "MESA-Web_Job_03242664908"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_report_data"
BOOTSTRAP_CACHE = PROJECT_ROOT / "data" / "processed" / "bootstrap_summary.json"
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


def _fmt_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _diagnostic_reason(row: dict, convective_delta: float | None) -> str:
    score = float(row["delta_global"])
    status = str(row["status"])
    if status == "Anomaly":
        return (
            f"The residual \\(\\delta_{{global}}={score:.3f}\\) exceeds the "
            f"\\(|\\delta_{{global}}|>{ANOMALY_THRESHOLD:.1f}\\) screening threshold, "
            "suggesting unresolved physics such as rotation, magnetism, tides, or "
            "another driver not included in the five-term model."
        )
    if convective_delta is not None and abs(convective_delta) > 0.25:
        return (
            "Deviation is accounted for by the convective driver "
            f"\\(\\Delta n_{{conv}}={convective_delta:.3f}\\). The superadiabatic "
            "surface layer is therefore treated as explained structure, not an "
            "unresolved anomaly."
        )
    return (
        f"The residual \\(\\delta_{{global}}={score:.3f}\\) remains below the "
        "screening threshold after subtracting the five physical deviation drivers."
    )


def _write_anomaly_screening(
    part3: Path,
    source_rows: list[dict],
    residual_rows: list[dict],
    deviation_rows: list[dict],
) -> None:
    sources = {row["star_id"]: row for row in source_rows}
    deviations = {row["star_id"]: row for row in deviation_rows}
    screening_rows = []
    for row in residual_rows:
        star_id = row["star_id"]
        source = sources.get(star_id, {})
        convective_delta = deviations.get(star_id, {}).get("delta_n_conv")
        classification = "Anomaly" if abs(float(row["delta_global"])) > ANOMALY_THRESHOLD else "Normal"
        screening_rows.append(
            {
                "star_profile_id": star_id,
                "mass_msun": source.get("mass_msun", ""),
                "age_gyr": float(source.get("age_years", 0.0)) / 1.0e9 if source else "",
                "n_global": row["n_global_observed"],
                "delta_global": row["delta_global"],
                "classification": classification,
                "diagnostic_reason": _diagnostic_reason({**row, "status": classification}, convective_delta),
            }
        )

    _write_csv(part3 / "anomaly_screening.csv", screening_rows)

    table_lines = [
        "| Star/Profile ID | Mass | Age | Global \\(n\\) | Anomaly Score \\(\\delta_{global}\\) | Classification | Diagnostic Reason |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for row in screening_rows:
        table_lines.append(
            "| {star_profile_id} | \\({mass}\\,M_\\odot\\) | {age} Gyr | {n_global} | {delta_global} | {classification} | {reason} |".format(
                star_profile_id=row["star_profile_id"],
                mass=_fmt_float(row["mass_msun"]),
                age=f"{float(row['age_gyr']):.3e}" if float(row["age_gyr"]) < 0.001 else _fmt_float(row["age_gyr"]),
                n_global=_fmt_float(row["n_global"]),
                delta_global=_fmt_float(row["delta_global"]),
                classification=row["classification"],
                reason=row["diagnostic_reason"],
            )
        )

    markdown = (
        "### 4.3 Anomaly Screening and Candidate Identification\n\n"
        "The anomaly screening stage was based on the master residual between the observed global "
        "polytropic structure and the amount of deviation explained by the five physical drivers. "
        "The anomaly score was defined as\n\n"
        "\\[\n"
        "\\delta_{global} = (n_{observed} - n_{base}) - \\sum \\langle \\Delta n_i \\rangle ,\n"
        "\\]\n\n"
        "where \\(n_{observed}\\) is the fitted global polytropic index, \\(n_{base}\\) is the expected "
        "reference polytropic index, and \\(\\sum \\langle \\Delta n_i \\rangle\\) is the combined "
        "contribution from radiation pressure, composition gradients, convection, nuclear energy "
        "generation, and degeneracy. A large local feature is therefore not automatically an anomaly. "
        "For example, the superadiabatic surface layer can produce a strong convective spike, but this "
        "feature is classified as normal when \\(\\Delta n_{conv}\\) accounts for it and leaves "
        "\\(\\delta_{global}\\approx0\\). In this implementation, a profile is screened as anomalous only "
        f"when \\(|\\delta_{{global}}|>{ANOMALY_THRESHOLD:.1f}\\), indicating residual structure beyond the "
        "noise threshold of the current five-driver model.\n\n"
        + "\n".join(table_lines)
        + "\n\n"
        "The current repository evidence contains the imported MESA-Web snapshots listed above. "
        "Additional 0.8, 2.0, and 5.0 \\(M_\\odot\\) tracks are treated as the next multitrack expansion "
        "targets until their profile files are imported and recorded in the source manifest. This keeps "
        "the results section consistent with the actual evidence package while preserving the planned "
        "screening logic for a broader stellar array.\n\n"
        "This screening demonstrates the purpose of the anomaly score: to find the needle in the "
        "haystack. Most stars can be structurally unusual during early evolution or near the surface, "
        "but they are still normal when the five physical drivers explain the deviation. A true "
        "candidate anomaly would appear only as a profile with a large \\(|\\delta_{global}|\\), meaning "
        "that known thermodynamic and structural corrections are insufficient. Future work will apply "
        "the same pipeline to the proposed full MIST grid of approximately 50,000 stellar models to "
        "isolate the small fraction of stars, potentially around 1%, whose residuals suggest missing "
        "physics such as magnetic fields, rapid rotation, binary tides, or other effects not yet "
        "included in the present driver model.\n"
    )
    (part3 / "section_4_3_anomaly_screening.md").write_text(markdown, encoding="utf-8")


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
        "method_pages": "18-42",
        "objectives_pages": "3-4",
    }
    if PAPER_REFERENCE.exists():
        record.update({"sha256": _sha256(PAPER_REFERENCE), "size_bytes": PAPER_REFERENCE.stat().st_size})
    return record


def _bootstrap_profile(task: tuple[str, int, int, int]) -> dict:
    job_text, profile_number, resamples, seed = task
    result = analyze_profile(load_mesa_web_job(Path(job_text), profile_number))
    profile = result["profile"]
    summary = bootstrap_global_n(
        np.asarray(profile["radius"]), np.asarray(profile["rho"]),
        float(result["input"]["teff"]), resamples, seed + profile_number,
    )
    summary.pop("samples")
    return {"track_id": Path(job_text).name, "profile": profile_number, **summary}


def _bootstrap_evidence(snapshots: list[dict], resamples: int, seed: int) -> list[dict]:
    if resamples > 0:
        tasks = [(str(item["_job"]), int(item["profile_number"]), resamples, seed) for item in snapshots]
        with ProcessPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
            rows = list(executor.map(_bootstrap_profile, tasks))
        _write_json(BOOTSTRAP_CACHE, {
            "jobs": sorted({_display_path(Path(item["_job"])) for item in snapshots}), "results": rows,
        })
        return rows
    if BOOTSTRAP_CACHE.exists():
        cached = json.loads(BOOTSTRAP_CACHE.read_text(encoding="utf-8"))
        expected = sorted({_display_path(Path(item["_job"])) for item in snapshots})
        cached_jobs = cached.get("jobs", [cached.get("job")])
        return cached["results"] if cached_jobs == expected else []
    return []


def build(
    job: Path,
    output: Path,
    bootstrap_resamples: int = 0,
    bootstrap_seed: int = 42,
    extra_jobs: list[Path] | None = None,
) -> None:
    _clean_output(output)
    (output / "README.md").write_text(
        """# Paper Report Data

This directory is generated from the legacy MESA-Web job by:

```powershell
.venv\\Scripts\\python.exe scripts\\build_paper_report_data.py
```

The numbered folders follow all of Chapter 3 on PDF pages 18-42 of the provided
research-paper reference. `METHOD_TRACEABILITY.csv` maps each method to its
evidence, and `MANIFEST.csv` records file sizes and SHA-256 checksums.

Important interpretation notes:

- The validation summary reports the actual number of tracks and profiles. It
  never substitutes evolutionary tables or synthetic profiles for MESA radial data.
- The PINN is not trained; no synthetic training result is presented as evidence.
- Hydrostatic failures are retained as findings rather than silently removed.
- Radiation uncertainty uses a recorded 1% input-error assumption. Seeded
  bootstrap summaries record completion and successful-fit rates per profile.
""",
        encoding="utf-8",
    )
    jobs = [job, *(extra_jobs or [])]
    snapshots = [dict(snapshot, _job=str(source_job)) for source_job in jobs for snapshot in list_mesa_profiles(source_job)]
    if not snapshots:
        raise ValueError("No MESA profiles found in the requested jobs")
    bootstrap_rows = _bootstrap_evidence(snapshots, bootstrap_resamples, bootstrap_seed)

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
    internal_consistency_rows: list[dict] = []
    radial_driver_rows: list[dict] = []

    for snapshot in snapshots:
        source_job = Path(snapshot["_job"])
        profile_number = int(snapshot["profile_number"])
        model = load_mesa_web_job(source_job, profile_number)
        result = analyze_profile(model)
        metadata = model.metadata
        profile = {key: np.asarray(value, dtype=float) for key, value in result["profile"].items()}
        star_id = f"mesa_model_{int(snapshot['model_number']):04d}_profile_{profile_number:02d}"

        source_path = Path(snapshot["path"])
        source_rows.append(
            {
                "star_id": star_id,
                "track_id": source_job.name,
                "source": "MESA-Web",
                "model_version": str(metadata.get("version_number", "unknown")),
                "profile_number": profile_number,
                "model_number": int(snapshot["model_number"]),
                "mass_msun": float(metadata["star_mass"]),
                "metallicity": float(metadata["initial_z"]),
                "age_years": float(metadata["star_age"]),
                "evolution_stage": "pre-main-sequence" if float(metadata["star_age"]) < 1e8 else "main-sequence",
                "priority": int(snapshot["priority"]),
                "source_file": _display_path(source_path),
                "size_bytes": source_path.stat().st_size,
                "sha256": _sha256(source_path),
                "usage_rights": "User-generated MESA-Web output for research use; cite MESA and MESA-Web",
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
        profile_path = output / "01_part1_polytropic_indices" / "profiles" / f"{star_id}.csv"
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
                "anomaly_threshold": result["residual"]["anomaly_threshold"],
                "status": result["status"],
            }
        )

        n_local = np.asarray(result["n_local"], dtype=float)
        finite_local = np.isfinite(n_local)
        mean_local = float(np.trapezoid(n_local[finite_local], profile["radius_fraction"][finite_local]))
        internal_consistency_rows.append(
            {
                "star_id": star_id,
                "n_global": result["global_fit"]["n_global"],
                "radial_mean_n_local": mean_local,
                "absolute_global_local_difference": abs(result["global_fit"]["n_global"] - mean_local),
                "chi2_global": result["global_fit"]["reduced_chi2"],
                "chi2_piecewise": result["piecewise_fit"]["chi2_piecewise"],
                "piecewise_improves_fit": result["piecewise_fit"]["chi2_piecewise"] < result["global_fit"]["reduced_chi2"],
            }
        )

        radius = profile["radius"]
        radius_fraction = profile["radius_fraction"]
        rho_profile = profile["rho"]
        temperature_profile = profile["temperature"]
        gas_pressure_profile = rho_profile * K_B_CGS * temperature_profile / (np.clip(profile["mu"], 0.1, None) * M_P_CGS)
        beta_profile = np.clip(gas_pressure_profile / np.maximum(profile["pressure"], 1e-99), 1e-6, 1.0)
        dlnrho_dr = np.gradient(np.log(np.clip(rho_profile, 1e-300, None)), radius, edge_order=2)
        h_rho = np.clip(1.0 / np.maximum(np.abs(dlnrho_dr), 1e-30), 0.0, np.nanmax(radius)) / np.nanmax(radius)
        tau_dyn = np.sqrt(np.maximum(radius**3 / (G_CGS * np.maximum(profile["mass_enclosed"], 1e-99)), 1e-30))
        tau_conv = 2.0 * tau_dyn * np.clip(1.0 - radius_fraction, 0.05, 1.0)
        radial_drivers = {
            "delta_n_rad": calculate_delta_n_rad(beta_profile, temperature_profile, rho_profile),
            "delta_n_mu": calculate_delta_n_mu(profile["mu"], radius_fraction, h_rho),
            "delta_n_conv": calculate_delta_n_conv(profile["grad_rad"], profile["grad_ad"], tau_conv, tau_dyn),
            "delta_n_deg": calculate_delta_n_deg(temperature_profile, rho_profile, X_fraction=0.70),
        }
        delta_n_nuc = calculate_delta_n_nuc(profile["epsilon"], rho_profile, radius)
        radial_driver_rows.extend(
            {
                "star_id": star_id,
                "radius_fraction": float(radius_fraction[index]),
                "beta": float(beta_profile[index]),
                "density_g_cm3": float(rho_profile[index]),
                "temperature_k": float(temperature_profile[index]),
                "mean_molecular_weight": float(profile["mu"][index]),
                "grad_ad": float(profile["grad_ad"][index]),
                "grad_rad": float(profile["grad_rad"][index]),
                "epsilon_erg_g_s": float(profile["epsilon"][index]),
                "delta_n_rad": float(radial_drivers["delta_n_rad"][index]),
                "delta_n_mu": float(radial_drivers["delta_n_mu"][index]),
                "delta_n_conv": float(radial_drivers["delta_n_conv"][index]),
                "delta_n_nuc_global": float(delta_n_nuc),
                "delta_n_deg": float(radial_drivers["delta_n_deg"][index]),
            }
            for index in range(len(radius_fraction))
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

    part1 = output / "01_part1_polytropic_indices"
    part2 = output / "02_part2_physical_deviations"
    part3 = output / "03_part3_global_comparison"
    part4 = output / "04_part4_computational_implementation"
    _write_csv(part1 / "source_manifest.csv", source_rows)
    _write_csv(part1 / "global_parameters.csv", global_rows)
    _write_csv(part1 / "quality_checks.csv", quality_rows)
    _write_csv(part1 / "global_fits.csv", global_fit_rows)
    _write_csv(part1 / "piecewise_fits.csv", piecewise_rows)
    _write_csv(part1 / "local_indices.csv", local_rows)
    _write_csv(part1 / "internal_consistency.csv", internal_consistency_rows)
    _write_csv(part2 / "radial_deviation_profiles.csv", radial_driver_rows)
    _write_csv(part2 / "deviation_summary.csv", deviation_rows)
    _write_csv(part2 / "radiation_error_propagation.csv", uncertainty_rows)
    _write_csv(part3 / "contribution_percentages.csv", contribution_rows)
    _write_csv(part3 / "global_residuals.csv", residual_rows)
    _write_anomaly_screening(part3, source_rows, residual_rows, deviation_rows)
    multitrack = assess_multitrack_manifest(source_rows)
    _write_json(
        part4 / "validation" / "multitrack_readiness.json",
        {**multitrack, "folds": leave_one_track_out([row["track_id"] for row in source_rows])},
    )

    _write_json(
        part1 / "bootstrap_protocol.json",
        {
            "method": "resample radial-density pairs with replacement and refit n_global",
            "required_resamples": 1000,
            "implemented_function": "stellar_analyzer.core.uncertainty.bootstrap_global_n",
            "status": "completed" if bootstrap_rows and all(row["valid"] for row in bootstrap_rows) else "incomplete",
            "minimum_success_rate": 0.90,
            "results": bootstrap_rows,
            "reproduction_command": ".venv\\Scripts\\python.exe scripts\\build_paper_report_data.py --bootstrap 1000 --seed 42",
        },
    )
    _write_json(
        part4 / "machine_learning" / "training_readiness.json",
        {
            "pinn_trained": False,
            "available_models": len(snapshots),
            "available_stellar_tracks": len(jobs),
            "paper_target_models": 50000,
            "architecture_implemented": True,
            "split_required": {"training_percent": 70, "validation_percent": 15, "test_percent": 15},
            "training_dataset": "data/processed/pinn_dataset.npz",
            "training_command": ".venv\\Scripts\\python.exe -m stellar_analyzer train-pinn --config configs\\pinn_training.json",
            "training_required": {"epochs": 300, "batch_size": 4, "learning_rate": 0.001, "patience": 30},
            "blocking_reason": "One solar-mass MESA track is not sufficient for a generalizable PINN.",
        },
    )
    _write_json(
        part4 / "storage" / "storage_evidence.json",
        {
            "interface": "command-line",
            "implementation": "stellar_analyzer.cli",
            "local_array_formats": ["CSV evidence package", "compressed NPZ training tensors", "PyTorch checkpoint"],
            "input_formats": ["MESA text profiles", "delimited text", "HDF5"],
            "database_required": False,
        },
    )
    _write_json(
        part4 / "validation" / "validation_summary.json",
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
            "external_multitrack_ready": multitrack["ready"],
            "external_multitrack_errors": multitrack["errors"],
            "test_command": ".venv\\Scripts\\python.exe -m pytest -q",
            "python_version": platform.python_version(),
        },
    )
    _write_json(
        part4 / "monitoring" / "reproducibility.json",
        {
            "builder": "scripts/build_paper_report_data.py",
            "source_jobs": [_display_path(source_job) for source_job in jobs],
            "deterministic_steps": "Steps 1-5 and 8-10",
            "stochastic_step": "Step 7 bootstrap; random seed must be recorded when run",
            "generated_file_count": sum(1 for path in output.rglob("*") if path.is_file()),
        },
    )

    traceability = [
        {"chapter_part": "Part 1", "paper_pages": "18-20", "method": "Lane-Emden RK4 and global n, K, alpha fitting", "evidence": "01_part1_polytropic_indices/global_fits.csv"},
        {"chapter_part": "Part 1", "paper_pages": "20-21", "method": "Local index with smoothing and center extrapolation", "evidence": "01_part1_polytropic_indices/local_indices.csv"},
        {"chapter_part": "Part 1", "paper_pages": "21-22", "method": "Core, radiative, and convective piecewise fitting", "evidence": "01_part1_polytropic_indices/piecewise_fits.csv"},
        {"chapter_part": "Part 1", "paper_pages": "22-23", "method": "Bootstrap and propagated uncertainty", "evidence": "01_part1_polytropic_indices/bootstrap_protocol.json; 02_part2_physical_deviations/radiation_error_propagation.csv"},
        {"chapter_part": "Part 1", "paper_pages": "23-24", "method": "Internal consistency and chi-square comparison", "evidence": "01_part1_polytropic_indices/internal_consistency.csv"},
        {"chapter_part": "Part 2", "paper_pages": "24-27", "method": "Radiation-pressure deviation", "evidence": "02_part2_physical_deviations/radial_deviation_profiles.csv"},
        {"chapter_part": "Part 2", "paper_pages": "27-29", "method": "Composition-gradient deviation", "evidence": "02_part2_physical_deviations/radial_deviation_profiles.csv"},
        {"chapter_part": "Part 2", "paper_pages": "29-30", "method": "Convective deviation", "evidence": "02_part2_physical_deviations/radial_deviation_profiles.csv"},
        {"chapter_part": "Part 2", "paper_pages": "30-31", "method": "Nuclear-concentration deviation", "evidence": "02_part2_physical_deviations/deviation_summary.csv"},
        {"chapter_part": "Part 2", "paper_pages": "31-35", "method": "Electron-degeneracy deviation", "evidence": "02_part2_physical_deviations/radial_deviation_profiles.csv"},
        {"chapter_part": "Part 3", "paper_pages": "36-37", "method": "Global residual and relative contributions", "evidence": "03_part3_global_comparison/"},
        {"chapter_part": "Section 4.3", "paper_pages": "results section", "method": "Anomaly screening using absolute global residual threshold", "evidence": "03_part3_global_comparison/section_4_3_anomaly_screening.md; 03_part3_global_comparison/anomaly_screening.csv"},
        {"chapter_part": "Part 4", "paper_pages": "37-40", "method": "Input, preprocessing, fitting, and driver modules", "evidence": "01_part1_polytropic_indices/; 02_part2_physical_deviations/"},
        {"chapter_part": "Part 4", "paper_pages": "40-41", "method": "PINN architecture and training protocol", "evidence": "04_part4_computational_implementation/machine_learning/training_readiness.json"},
        {"chapter_part": "Part 4", "paper_pages": "41-42", "method": "Storage, validation, and monitoring", "evidence": "04_part4_computational_implementation/"},
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
    parser.add_argument("--extra-job", type=Path, action="append", default=[], help="additional MESA-Web track directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=0, metavar="RESAMPLES")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.job, args.output, args.bootstrap, args.seed, args.extra_job)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
