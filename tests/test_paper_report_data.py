import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stellar_analyzer.core.validation import assess_multitrack_manifest, leave_one_track_out


ROOT = Path(__file__).parents[1]
REPORT = ROOT / "paper_report_data"


def test_paper_report_data_is_complete_and_consistent():
    part1 = REPORT / "01_part1_polytropic_indices"
    part2 = REPORT / "02_part2_physical_deviations"
    part3 = REPORT / "03_part3_global_comparison"
    sources = pd.read_csv(part1 / "source_manifest.csv")
    parameters = pd.read_csv(part1 / "global_parameters.csv")
    local = pd.read_csv(part1 / "local_indices.csv")
    contributions = pd.read_csv(part3 / "contribution_percentages.csv")
    residuals = pd.read_csv(part3 / "global_residuals.csv")
    anomaly_screening = pd.read_csv(part3 / "anomaly_screening.csv")
    radial_drivers = pd.read_csv(part2 / "radial_deviation_profiles.csv")

    assert len(sources) == len(parameters) == len(residuals) == 8
    assert len(anomaly_screening) == len(residuals)
    assert residuals["anomaly_threshold"].eq(5.0).all()
    assert anomaly_screening["classification"].eq("Normal").all()
    assert (part3 / "section_4_3_anomaly_screening.md").read_text(encoding="utf-8").startswith(
        "### 4.3 Anomaly Screening and Candidate Identification"
    )
    assert parameters["n_points_processed"].eq(500).all()
    assert len(local) == 8 * 500
    assert len(radial_drivers) == 8 * 500
    assert contributions.groupby("star_id")["absolute_contribution_percent"].sum().pipe(
        lambda values: np.allclose(values, 100.0)
    )

    readiness = json.loads((REPORT / "04_part4_computational_implementation" / "machine_learning" / "training_readiness.json").read_text(encoding="utf-8"))
    assert readiness["pinn_trained"] is False
    assert readiness["available_models"] == 8


def test_paper_report_manifest_hashes_match():
    manifest = pd.read_csv(REPORT / "MANIFEST.csv")
    for row in manifest.itertuples(index=False):
        path = REPORT / row.relative_path
        assert path.is_file()
        assert path.stat().st_size == row.size_bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256


def test_leave_one_track_out_never_leaks_profiles():
    track_ids = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
    folds = leave_one_track_out(track_ids)
    assert len(folds) == 4
    for fold in folds:
        assert all(track_ids[index] != fold["held_out_track"] for index in fold["train_indices"])
        assert all(track_ids[index] == fold["held_out_track"] for index in fold["validation_indices"])


def test_multitrack_gate_requires_real_coverage_and_provenance():
    rows = []
    for track, mass in zip("abcd", (0.8, 1.0, 2.0, 5.0)):
        for age in (1.0, 2.0, 3.0):
            rows.append({
                "track_id": track, "source": "MESA", "model_version": "test",
                "mass_msun": mass, "metallicity": 0.02, "age_years": age,
                "evolution_stage": "test", "sha256": "abc", "usage_rights": "test data",
            })
    assert assess_multitrack_manifest(rows)["ready"] is True
    assert assess_multitrack_manifest(rows[:3])["ready"] is False
