"""Tests for Phase 8 external validation (Model S + Kepler LEGACY)."""

from pathlib import Path

import numpy as np
import pytest

from stellar_analyzer.core.external_validation import (
    DEFAULT_KEPLER_LEGACY_PATH,
    DEFAULT_MODEL_S_PATH,
    load_kepler_legacy_sample,
    load_ssm_profile,
    validate_against_kepler_legacy,
    validate_against_ssm,
    validate_pipeline_result,
)
from stellar_analyzer.core.global_fit import fit_global_polytrope


def test_bundled_reference_files_exist():
    assert DEFAULT_MODEL_S_PATH.exists()
    assert DEFAULT_KEPLER_LEGACY_PATH.exists()


def test_load_ssm_profile_is_sorted_and_physically_sane():
    ssm = load_ssm_profile()
    assert np.all(np.diff(ssm["r_over_r"]) > 0)
    # Density must decrease monotonically outward in a real stellar model.
    assert np.all(np.diff(ssm["rho"]) <= 1e-6)
    assert ssm["rho"][0] > 100.0  # central density should be of order ~150 g/cm^3
    assert ssm["temperature"][0] > 1.0e7  # central temperature should be of order ~1.57e7 K


def test_load_kepler_legacy_sample_has_expected_columns_and_real_kic():
    sample = load_kepler_legacy_sample()
    assert len(sample) >= 4
    assert 2837475 in sample["kic"].to_numpy()  # a real, published LEGACY target


def test_validate_against_ssm_matches_itself_almost_perfectly():
    """Fitting the SSM profile against itself must show near-zero Delta n/RMS and PSI near 1."""

    ssm = load_ssm_profile()
    ssm_fit = fit_global_polytrope(ssm["r_over_r"], ssm["rho"], 5778.0)

    result = validate_against_ssm(
        r_over_r=ssm["r_over_r"],
        rho=ssm["rho"],
        n_global_computed=ssm_fit.n_global,
        pressure=ssm["pressure"],
    )

    assert result.reference == "model_s"
    assert abs(result.delta_n) < 1e-6
    assert result.rms < 0.05
    assert result.psi > 0.9
    assert result.consistent is True


def test_validate_against_ssm_detects_a_grossly_different_profile():
    ssm = load_ssm_profile()
    wildly_different_rho = ssm["rho"] * 50.0 + 1.0
    result = validate_against_ssm(
        r_over_r=ssm["r_over_r"],
        rho=wildly_different_rho,
        n_global_computed=ssm_fit_n(ssm, wildly_different_rho),
    )
    assert result.psi < 0.9
    assert result.rms > 0.1


def ssm_fit_n(ssm, rho):
    return fit_global_polytrope(ssm["r_over_r"], rho, 5778.0).n_global


def test_validate_against_ssm_raises_on_non_overlapping_grid():
    ssm = load_ssm_profile()
    with pytest.raises(ValueError):
        validate_against_ssm(
            r_over_r=np.array([5.0, 6.0, 7.0]),
            rho=np.array([1.0, 1.0, 1.0]),
            n_global_computed=1.5,
        )


def test_validate_against_kepler_legacy_exact_match_by_kic():
    result = validate_against_kepler_legacy(mass_msun=1.42, teff_k=6614, age_gyr=1.59, kic=2837475)
    assert result.reference == "kepler_legacy"
    assert result.details["matched_kic"] == 2837475
    assert result.consistent is True
    assert abs(result.delta_n) < 0.01  # mass offset should be ~0 for an exact match


def test_validate_against_kepler_legacy_nearest_teff_when_kic_not_given():
    result = validate_against_kepler_legacy(mass_msun=1.4, teff_k=6610, age_gyr=1.6)
    assert result.details["matched_kic"] == 2837475  # closest Teff in the bundled sample


def test_validate_against_kepler_legacy_flags_gross_mismatch():
    result = validate_against_kepler_legacy(mass_msun=20.0, teff_k=6614, age_gyr=1.59, kic=2837475)
    assert result.consistent is False


def test_validate_against_kepler_legacy_unknown_kic_raises():
    with pytest.raises(ValueError):
        validate_against_kepler_legacy(mass_msun=1.0, teff_k=5778, age_gyr=4.6, kic=999999999)


def test_validate_pipeline_result_shape():
    from stellar_analyzer.core.pipeline import analyze_star

    result = analyze_star({"mass": 1.0, "teff": 5778.0, "metallicity": 0.0, "age": 4.6})
    report = validate_pipeline_result(result)
    assert set(report) == {"model_s", "kepler_legacy"}
    assert "delta_n" in report["model_s"]
    assert "delta_n" in report["kepler_legacy"]
