import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
    radial_drivers = pd.read_csv(part2 / "radial_deviation_profiles.csv")

    assert len(sources) == len(parameters) == len(residuals) == 8
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
