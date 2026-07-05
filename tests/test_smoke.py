from stellar_analyzer import analyze_star, batch_analyze
from stellar_analyzer.core.data_loader import list_mesa_profiles, load_mesa_web_job
from stellar_analyzer.core.deviation_drivers import calculate_delta_n_conv
from stellar_analyzer.core.constants import G_CGS
from stellar_analyzer.core.global_fit import fit_global_polytrope, solve_lane_emden_rk4
from stellar_analyzer.core.pipeline import analyze_mesa_job
from stellar_analyzer.core.piecewise_fit import fit_piecewise
from stellar_analyzer.core.preprocess import preprocess_profile

import numpy as np

import pandas as pd
from pathlib import Path


def test_single_star_analysis_smoke():
    result = analyze_star({"name": "Sun", "mass": 1.0, "teff": 5778.0, "metallicity": 0.0, "age": 4.6})
    assert result["global_fit"]["n_global"] > 0
    assert result["global_fit"]["rho_c"] > 0
    assert result["global_fit"]["K"] > 0
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


def test_convection_correction_is_bounded_for_extreme_surface_gradients():
    correction = calculate_delta_n_conv(
        np.array([0.5, 1e9]), np.array([0.4, 0.4]), np.array([2.0, 1e9]), np.ones(2)
    )
    assert np.all(np.isfinite(correction))
    assert correction.max() <= 12.0


def test_young_mesa_profile_has_reasonable_residual_and_hydrostatic_check():
    job = Path(__file__).parents[1] / "data" / "raw" / "MESA-Web_Job_03242664908"
    result = analyze_mesa_job(job, profile_number=2)
    assert -5.0 <= result["anomaly_score"] <= 5.0
    assert result["deviation_factors"]["delta_n_conv"] <= 12.0
    assert result["preprocessing"]["hydrostatic_ok"] is True
    assert result["piecewise_fit"]["success"] is True
    assert max(result["piecewise_fit"]["continuity_errors"].values()) < 1e-6


def test_global_fit_recovers_physical_polytrope_parameters():
    n_true = 1.5
    rho_c_true = 120.0
    alpha_true = 8.0e9
    xi, theta = solve_lane_emden_rk4(n_true, xi_max=3.2, step=0.01)
    radius = xi * alpha_true
    density = rho_c_true * theta**n_true

    fit = fit_global_polytrope(radius, density, Teff=5778.0)
    expected_k = 4.0 * np.pi * G_CGS * alpha_true**2 * rho_c_true ** (1.0 - 1.0 / n_true) / (n_true + 1.0)

    assert np.isclose(fit.n_global, n_true, rtol=0.03)
    assert np.isclose(fit.rho_c, rho_c_true, rtol=0.03)
    assert np.isclose(fit.alpha, alpha_true, rtol=0.03)
    assert np.isclose(fit.K, expected_k, rtol=0.08)


def test_piecewise_fit_rejects_undersampled_profiles():
    radius = np.linspace(0.0, 1.0, 11)
    with np.testing.assert_raises_regex(ValueError, "at least 12"):
        fit_piecewise(radius, np.ones(11), np.ones(11), np.ones(11), np.ones(11))
