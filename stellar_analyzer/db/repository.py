"""Insert/query functions for the PostgreSQL storage module.

Built on the schema in :mod:`stellar_analyzer.db.schema` and the
compression/chunking helpers in :mod:`stellar_analyzer.db.binary_store`.
Every function here takes an open psycopg2 ``connection`` so callers (and
tests) control connection lifecycle and transactions explicitly.
"""

from __future__ import annotations

import numpy as np

from stellar_analyzer.db.binary_store import ArrayBlob, compress_array, decompress_array


def insert_star(connection, star: dict) -> int:
    """Insert one Level-1 global-parameter row; returns the new ``star_id``."""

    columns = (
        "name", "source", "mass_msun", "radius_rsun", "teff_k", "luminosity_lsun",
        "metallicity_feh", "age_gyr", "evolution_stage", "n_points",
    )
    values = tuple(star.get(column) for column in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    with connection.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO stars ({', '.join(columns)}) VALUES ({placeholders}) RETURNING star_id",
            values,
        )
        star_id = cursor.fetchone()[0]
    connection.commit()
    return int(star_id)


def insert_radial_profile(connection, star_id: int, profile: dict[str, np.ndarray]) -> int:
    """Insert one star's Level-2 radial-profile rows (one row per sample)."""

    columns = (
        "r_over_r", "rho_g_cm3", "pressure_dyn_cm2", "temperature_k",
        "x_fraction", "y_fraction", "z_fraction", "epsilon", "grad_ad", "grad_rad",
    )
    length = len(np.asarray(profile["r_over_r"]))
    rows = []
    for index in range(length):
        row = [star_id]
        for column in columns:
            array = profile.get(column)
            row.append(float(np.asarray(array)[index]) if array is not None else None)
        rows.append(tuple(row))

    placeholders = ", ".join(["%s"] * (len(columns) + 1))
    with connection.cursor() as cursor:
        cursor.executemany(
            f"INSERT INTO radial_profiles (star_id, {', '.join(columns)}) VALUES ({placeholders})",
            rows,
        )
    connection.commit()
    return length


def insert_binary_array(connection, star_id: int, array_name: str, array: np.ndarray) -> ArrayBlob:
    """Compress, chunk, and store ``array`` under BYTEA rows (step 8.3)."""

    blob = compress_array(array)
    rows = [
        (star_id, array_name, blob.dtype, list(blob.shape), blob.compression, index, blob.chunk_count, chunk)
        for index, chunk in enumerate(blob.chunks)
    ]
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM profile_blobs WHERE star_id = %s AND array_name = %s", (star_id, array_name))
        cursor.executemany(
            "INSERT INTO profile_blobs "
            "(star_id, array_name, dtype, shape, compression, chunk_index, chunk_count, chunk_data) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            rows,
        )
    connection.commit()
    return blob


def fetch_binary_array(connection, star_id: int, array_name: str) -> np.ndarray:
    """Fetch and reassemble a stored array (full read; see also ``fetch_binary_array_prefix``)."""

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT dtype, shape, compression, chunk_data FROM profile_blobs "
            "WHERE star_id = %s AND array_name = %s ORDER BY chunk_index",
            (star_id, array_name),
        )
        rows = cursor.fetchall()
    if not rows:
        raise KeyError(f"No stored array '{array_name}' for star_id={star_id}")
    dtype, shape, compression, _ = rows[0]
    chunks = [row[3] for row in rows]
    blob = ArrayBlob(dtype=dtype, shape=tuple(shape), compression=compression, chunks=[bytes(chunk) for chunk in chunks])
    return decompress_array(blob)


def find_stars_by_parameter_range(
    connection,
    mass_range: tuple[float, float] | None = None,
    teff_range: tuple[float, float] | None = None,
    metallicity_range: tuple[float, float] | None = None,
    age_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Search stars by global-parameter ranges (step 8.2's indexed search)."""

    clauses, params = [], []
    for column, bounds in (
        ("mass_msun", mass_range),
        ("teff_k", teff_range),
        ("metallicity_feh", metallicity_range),
        ("age_gyr", age_range),
    ):
        if bounds is not None:
            clauses.append(f"{column} BETWEEN %s AND %s")
            params.extend(bounds)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT star_id, name, source, mass_msun, radius_rsun, teff_k, luminosity_lsun, "
            f"metallicity_feh, age_gyr, evolution_stage, n_points FROM stars {where} ORDER BY star_id",
            params,
        )
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
