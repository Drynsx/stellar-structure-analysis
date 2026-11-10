"""Tests for the Phase 9 PostgreSQL storage module (Chapter 3, steps 8-9).

Three tiers, matching step 9's own breakdown:

* 9.1 Unit tests -- pure compression/chunking functions, no DB needed.
* 9.2 Integration tests -- schema creation and insert/fetch against a live
  PostgreSQL instance. Skipped cleanly when none is reachable.
* 9.3 Validation tests -- round-trip integrity (what goes in is what comes
  back out, bit-for-bit) against a live instance. Also skipped cleanly.

Point a live PostgreSQL instance at this suite with
``STELLAR_ANALYZER_DATABASE_URL=postgresql://user:pass@host:5432/db``.
"""

from __future__ import annotations

import numpy as np
import pytest

from stellar_analyzer.db.binary_store import (
    DEFAULT_CHUNK_SIZE,
    chunk_bytes,
    compress_array,
    decompress_array,
    partial_decompress_prefix,
    reassemble_chunks,
)

psycopg2 = pytest.importorskip("psycopg2")

from stellar_analyzer.db import repository, schema  # noqa: E402
from stellar_analyzer.db.connection import connect, is_available  # noqa: E402


# --------------------------------------------------------------------------
# 9.1 Unit tests: pure functions, no database required.
# --------------------------------------------------------------------------


def test_chunk_bytes_and_reassemble_round_trip():
    data = bytes(range(256)) * 1000  # 256 KiB
    chunks = chunk_bytes(data, chunk_size=1024)
    assert len(chunks) == len(data) // 1024
    assert reassemble_chunks(chunks) == data


def test_chunk_bytes_handles_empty_input():
    assert chunk_bytes(b"") == [b""]
    assert reassemble_chunks(chunk_bytes(b"")) == b""


def test_chunk_bytes_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        chunk_bytes(b"data", chunk_size=0)


def test_compress_array_round_trips_exactly_for_various_dtypes():
    for dtype in (np.float64, np.float32, np.int32):
        array = np.linspace(0.0, 1.0, 5000).astype(dtype)
        blob = compress_array(array, chunk_size=4096)
        restored = decompress_array(blob)
        np.testing.assert_array_equal(restored, array)
        assert restored.dtype == array.dtype
        assert restored.shape == array.shape


def test_compress_array_preserves_multidimensional_shape():
    array = np.random.default_rng(0).normal(size=(37, 5)).astype(np.float64)
    blob = compress_array(array)
    restored = decompress_array(blob)
    np.testing.assert_allclose(restored, array)
    assert restored.shape == (37, 5)


def test_compress_array_actually_shrinks_a_compressible_array():
    array = np.zeros(100_000, dtype=np.float64)  # highly compressible
    blob = compress_array(array)
    assert blob.compressed_size < array.nbytes // 2


def test_compress_array_chunking_matches_default_chunk_size():
    array = np.arange(3 * DEFAULT_CHUNK_SIZE, dtype=np.uint8)
    blob = compress_array(array)
    assert blob.chunk_count >= 1
    assert all(len(chunk) <= DEFAULT_CHUNK_SIZE for chunk in blob.chunks)


def test_decompress_array_rejects_unknown_compression():
    array = np.arange(10, dtype=np.float64)
    blob = compress_array(array)
    bad_blob = blob.__class__(dtype=blob.dtype, shape=blob.shape, compression="lz4", chunks=blob.chunks)
    with pytest.raises(ValueError):
        decompress_array(bad_blob)


def test_partial_decompress_prefix_is_a_prefix_of_the_full_stream():
    array = np.arange(5000, dtype=np.float64)
    blob = compress_array(array, chunk_size=256)
    full = reassemble_chunks(blob.chunks)
    prefix = partial_decompress_prefix(blob, chunk_limit=2)
    assert full.startswith(prefix)


# --------------------------------------------------------------------------
# 9.2 / 9.3 Integration + validation tests: require a live PostgreSQL DB.
# --------------------------------------------------------------------------

DB_AVAILABLE = is_available()
skip_without_db = pytest.mark.skipif(not DB_AVAILABLE, reason="no live PostgreSQL instance reachable")


@pytest.fixture
def db_connection():
    connection = connect()
    schema.create_schema(connection)
    try:
        yield connection
    finally:
        schema.drop_schema(connection)
        connection.close()


@skip_without_db
def test_create_schema_is_idempotent(db_connection):
    schema.create_schema(db_connection)  # must not raise on a second call
    with db_connection.cursor() as cursor:
        cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        tables = {row[0] for row in cursor.fetchall()}
    assert {"stars", "radial_profiles", "derived_quantities", "profile_blobs"}.issubset(tables)


@skip_without_db
def test_insert_and_query_star_round_trips_global_parameters(db_connection):
    star_id = repository.insert_star(
        db_connection,
        {
            "name": "Test Star", "source": "unit-test", "mass_msun": 1.0, "radius_rsun": 1.0,
            "teff_k": 5778.0, "luminosity_lsun": 1.0, "metallicity_feh": 0.0, "age_gyr": 4.6,
            "evolution_stage": "main-sequence", "n_points": 500,
        },
    )
    assert isinstance(star_id, int)

    matches = repository.find_stars_by_parameter_range(db_connection, mass_range=(0.9, 1.1))
    assert any(row["star_id"] == star_id for row in matches)
    matches_outside = repository.find_stars_by_parameter_range(db_connection, mass_range=(5.0, 10.0))
    assert all(row["star_id"] != star_id for row in matches_outside)


@skip_without_db
def test_insert_radial_profile_round_trips_row_count(db_connection):
    star_id = repository.insert_star(
        db_connection,
        {"name": "Profile Star", "source": "unit-test", "mass_msun": 1.0, "teff_k": 5778.0,
         "metallicity_feh": 0.0, "age_gyr": 4.6},
    )
    r_over_r = np.linspace(0.0, 1.0, 50)
    profile = {"r_over_r": r_over_r, "rho_g_cm3": 150.0 * (1.0 - r_over_r) ** 2 + 1e-6,
               "pressure_dyn_cm2": 1e17 * (1.0 - r_over_r) ** 3 + 1.0}
    inserted = repository.insert_radial_profile(db_connection, star_id, profile)
    assert inserted == 50
    with db_connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM radial_profiles WHERE star_id = %s", (star_id,))
        assert cursor.fetchone()[0] == 50


@skip_without_db
def test_insert_and_fetch_binary_array_round_trips_bit_for_bit(db_connection):
    star_id = repository.insert_star(
        db_connection,
        {"name": "Blob Star", "source": "unit-test", "mass_msun": 1.0, "teff_k": 5778.0,
         "metallicity_feh": 0.0, "age_gyr": 4.6},
    )
    array = np.random.default_rng(42).normal(size=10_000).astype(np.float64)
    repository.insert_binary_array(db_connection, star_id, "density_profile", array)

    restored = repository.fetch_binary_array(db_connection, star_id, "density_profile")
    np.testing.assert_array_equal(restored, array)


@skip_without_db
def test_fetch_binary_array_missing_raises_keyerror(db_connection):
    star_id = repository.insert_star(
        db_connection,
        {"name": "Empty Star", "source": "unit-test", "mass_msun": 1.0, "teff_k": 5778.0,
         "metallicity_feh": 0.0, "age_gyr": 4.6},
    )
    with pytest.raises(KeyError):
        repository.fetch_binary_array(db_connection, star_id, "does_not_exist")
