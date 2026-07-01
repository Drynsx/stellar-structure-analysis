import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]
REPORT = ROOT / "paper_report_data"


def test_paper_report_data_is_complete_and_consistent():
    sources = pd.read_csv(REPORT / "01_input" / "source_manifest.csv")
    parameters = pd.read_csv(REPORT / "01_input" / "global_parameters.csv")
    local = pd.read_csv(REPORT / "03_polytropic_indices" / "local_indices.csv")
    contributions = pd.read_csv(REPORT / "04_deviation_drivers" / "contribution_percentages.csv")
    residuals = pd.read_csv(REPORT / "05_global_comparison" / "global_residuals.csv")

    assert len(sources) == len(parameters) == len(residuals) == 8
    assert parameters["n_points_processed"].eq(500).all()
    assert len(local) == 8 * 500
    assert contributions.groupby("star_id")["absolute_contribution_percent"].sum().pipe(
        lambda values: np.allclose(values, 100.0)
    )

    readiness = json.loads((REPORT / "07_machine_learning" / "training_readiness.json").read_text(encoding="utf-8"))
    assert readiness["pinn_trained"] is False
    assert readiness["available_models"] == 8


def test_paper_report_manifest_hashes_match():
    manifest = pd.read_csv(REPORT / "MANIFEST.csv")
    for row in manifest.itertuples(index=False):
        path = REPORT / row.relative_path
        assert path.is_file()
        assert path.stat().st_size == row.size_bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256
