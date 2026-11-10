"""Synthetic round-trip tests for the Phase 7 FITS/VOTable readers.

These skip cleanly when astropy is not installed, so the rest of the suite
keeps passing on machines that haven't opted into the optional 'fits' extra.
"""

import numpy as np
import pytest

astropy = pytest.importorskip("astropy")

from stellar_analyzer.core.data_loader import (  # noqa: E402
    read_fits_asteroseismic_catalog,
    read_votable_benchmark_catalog,
)


SYNTHETIC_ROWS = {
    "KIC": np.array([1435467, 6116048, 7871531], dtype=np.int64),
    "numax": np.array([1406.0, 1746.0, 3426.0], dtype=np.float64),
    "Dnu": np.array([83.5, 98.0, 155.4], dtype=np.float64),
    "Teff": np.array([6326.0, 6033.0, 5488.0], dtype=np.float64),
    "mass": np.array([1.47, 1.08, 0.99], dtype=np.float64),
    "radius": np.array([2.21, 1.22, 0.96], dtype=np.float64),
}


def test_fits_asteroseismic_catalog_round_trips(tmp_path):
    from astropy.io import fits

    columns = [
        fits.Column(name="KIC", array=SYNTHETIC_ROWS["KIC"], format="K"),
        fits.Column(name="numax", array=SYNTHETIC_ROWS["numax"], format="D"),
        fits.Column(name="Dnu", array=SYNTHETIC_ROWS["Dnu"], format="D"),
        fits.Column(name="Teff", array=SYNTHETIC_ROWS["Teff"], format="D"),
        fits.Column(name="mass", array=SYNTHETIC_ROWS["mass"], format="D"),
        fits.Column(name="radius", array=SYNTHETIC_ROWS["radius"], format="D"),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    path = tmp_path / "synthetic_kepler_asteroseismic.fits"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    frame = read_fits_asteroseismic_catalog(path)

    assert len(frame) == 3
    assert "star_id" in frame.columns  # KIC -> star_id
    assert "numax" in frame.columns
    assert "delta_nu" in frame.columns  # Dnu -> delta_nu
    assert "teff" in frame.columns
    assert "mass_msun" in frame.columns
    assert "radius_rsun" in frame.columns
    np.testing.assert_allclose(sorted(frame["star_id"].tolist()), sorted(SYNTHETIC_ROWS["KIC"].tolist()))
    np.testing.assert_allclose(sorted(frame["numax"].tolist()), sorted(SYNTHETIC_ROWS["numax"].tolist()))


def test_fits_asteroseismic_catalog_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_fits_asteroseismic_catalog(tmp_path / "does_not_exist.fits")


def test_votable_benchmark_catalog_round_trips(tmp_path):
    from astropy.table import Table

    table = Table(
        {
            "source_id": np.array([4472832130942575872, 3864972938605115520], dtype=np.int64),
            "teff": np.array([5778.0, 4830.0], dtype=np.float64),
            "feh": np.array([0.0, -0.3], dtype=np.float64),
            "logg": np.array([4.44, 4.6], dtype=np.float64),
            "mass": np.array([1.0, 0.75], dtype=np.float64),
            "radius": np.array([1.0, 0.72], dtype=np.float64),
        }
    )
    path = tmp_path / "synthetic_gaia_benchmark.vot"
    table.write(path, format="votable")

    frame = read_votable_benchmark_catalog(path)

    assert len(frame) == 2
    assert "star_id" in frame.columns  # source_id -> star_id
    assert "teff" in frame.columns
    assert "metallicity" in frame.columns  # feh -> metallicity
    assert "mass_msun" in frame.columns
    assert "radius_rsun" in frame.columns
    np.testing.assert_allclose(sorted(frame["teff"].tolist()), [4830.0, 5778.0])


def test_votable_benchmark_catalog_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_votable_benchmark_catalog(tmp_path / "does_not_exist.vot")
