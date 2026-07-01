from stellar_analyzer import analyze_star, batch_analyze
from stellar_analyzer.core.data_loader import list_mesa_profiles, load_mesa_web_job
from stellar_analyzer.core.preprocess import preprocess_profile

import pandas as pd
from pathlib import Path


def test_single_star_analysis_smoke():
    result = analyze_star({"name": "Sun", "mass": 1.0, "teff": 5778.0, "metallicity": 0.0, "age": 4.6})
    assert result["global_fit"]["n_global"] > 0
    assert set(result["deviation_factors"]) == {
        "delta_n_rad",
        "delta_n_mu",
        "delta_n_conv",
        "delta_n_nuc",
        "delta_n_deg",
    }


def test_batch_analysis_smoke():
    frame = pd.DataFrame(
        [
            {"mass": 1.0, "teff": 5778.0, "metallicity": 0.0, "age": 4.6},
            {"mass": 2.0, "teff": 9000.0, "metallicity": 0.2, "age": 0.5},
        ]
    )
    result = batch_analyze(frame)
    assert "anomaly_score" in result.columns
    assert len(result) == 2


def test_legacy_mesa_web_job_loads_and_normalizes():
    job = Path(__file__).parents[1] / "data" / "raw" / "MESA-Web_Job_03242664908"
    snapshots = list_mesa_profiles(job)
    assert [item["profile_number"] for item in snapshots] == list(range(1, 9))

    model = load_mesa_web_job(job, profile_number=8)
    assert model.metadata["model_number"] == 295
    assert {"radius", "rho", "pressure", "temperature", "mass_enclosed"}.issubset(model.arrays)

    prepared = preprocess_profile(model)
    assert len(prepared.profile["radius"]) == 500
    assert prepared.profile["radius"][0] < prepared.profile["radius"][-1]
    assert prepared.profile["mass_enclosed"][-1] > 1e33
