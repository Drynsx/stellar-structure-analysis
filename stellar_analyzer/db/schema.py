"""PostgreSQL schema (Chapter 3, step 8.1) and indexing (step 8.2).

The three tables below mirror the three data levels defined in step 1 of
Chapter 3 ("การจัดการข้อมูลนำเข้า"):

* Level 1 (global parameters): ``stars`` -- mass, radius, Teff, luminosity,
  metallicity, age, evolution stage, and point count per star.
* Level 2 (radial profiles): ``radial_profiles`` -- one row per
  (star, radius) sample: r/R, density, pressure, temperature, composition
  (X, Y, Z), energy generation rate, and the adiabatic/radiative
  gradients.
* Level 3 (derived quantities): ``derived_quantities`` -- one row per
  (star, radius) sample of derivative/scale-height/physics-parameter
  columns (d(rho)/dr, d(P)/dr, pressure scale height, beta, eta, ...).

``profile_blobs`` is the BYTEA-backed binary store for step 8.3: each row
is one gzip(6)-compressed, chunked array (see
:mod:`stellar_analyzer.db.binary_store`), keyed by star and array name.
"""

from __future__ import annotations

CREATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stars (
    star_id             BIGSERIAL PRIMARY KEY,
    name                TEXT NOT NULL,
    source              TEXT NOT NULL DEFAULT 'unknown',
    mass_msun           DOUBLE PRECISION NOT NULL,
    radius_rsun         DOUBLE PRECISION,
    teff_k              DOUBLE PRECISION NOT NULL,
    luminosity_lsun     DOUBLE PRECISION,
    metallicity_feh     DOUBLE PRECISION NOT NULL,
    age_gyr             DOUBLE PRECISION NOT NULL,
    evolution_stage     TEXT,
    n_points            INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS radial_profiles (
    id                  BIGSERIAL PRIMARY KEY,
    star_id             BIGINT NOT NULL REFERENCES stars(star_id) ON DELETE CASCADE,
    r_over_r            DOUBLE PRECISION NOT NULL,
    rho_g_cm3           DOUBLE PRECISION NOT NULL,
    pressure_dyn_cm2    DOUBLE PRECISION NOT NULL,
    temperature_k       DOUBLE PRECISION,
    x_fraction          DOUBLE PRECISION,
    y_fraction          DOUBLE PRECISION,
    z_fraction          DOUBLE PRECISION,
    epsilon             DOUBLE PRECISION,
    grad_ad             DOUBLE PRECISION,
    grad_rad            DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS derived_quantities (
    id                  BIGSERIAL PRIMARY KEY,
    star_id             BIGINT NOT NULL REFERENCES stars(star_id) ON DELETE CASCADE,
    r_over_r            DOUBLE PRECISION NOT NULL,
    drho_dr             DOUBLE PRECISION,
    dpressure_dr        DOUBLE PRECISION,
    pressure_scale_height DOUBLE PRECISION,
    beta                DOUBLE PRECISION,
    eta                 DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS profile_blobs (
    id                  BIGSERIAL PRIMARY KEY,
    star_id             BIGINT NOT NULL REFERENCES stars(star_id) ON DELETE CASCADE,
    array_name          TEXT NOT NULL,
    dtype               TEXT NOT NULL,
    shape               INTEGER[] NOT NULL,
    compression         TEXT NOT NULL DEFAULT 'gzip',
    chunk_index         INTEGER NOT NULL,
    chunk_count         INTEGER NOT NULL,
    chunk_data          BYTEA NOT NULL,
    UNIQUE (star_id, array_name, chunk_index)
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_stars_mass ON stars (mass_msun);
CREATE INDEX IF NOT EXISTS idx_stars_teff ON stars (teff_k);
CREATE INDEX IF NOT EXISTS idx_stars_metallicity ON stars (metallicity_feh);
CREATE INDEX IF NOT EXISTS idx_stars_age ON stars (age_gyr);
CREATE INDEX IF NOT EXISTS idx_stars_source ON stars (source);

CREATE INDEX IF NOT EXISTS idx_radial_profiles_star_id ON radial_profiles (star_id);
CREATE INDEX IF NOT EXISTS idx_radial_profiles_star_r ON radial_profiles (star_id, r_over_r);

CREATE INDEX IF NOT EXISTS idx_derived_quantities_star_id ON derived_quantities (star_id);
CREATE INDEX IF NOT EXISTS idx_derived_quantities_star_r ON derived_quantities (star_id, r_over_r);

CREATE INDEX IF NOT EXISTS idx_profile_blobs_star_array ON profile_blobs (star_id, array_name);
"""

DROP_SCHEMA_SQL = """
DROP TABLE IF EXISTS profile_blobs;
DROP TABLE IF EXISTS derived_quantities;
DROP TABLE IF EXISTS radial_profiles;
DROP TABLE IF EXISTS stars;
"""


def create_schema(connection) -> None:
    """Create the schema and its indexes (steps 8.1 and 8.2) if not present."""

    with connection.cursor() as cursor:
        cursor.execute(CREATE_SCHEMA_SQL)
        cursor.execute(CREATE_INDEXES_SQL)
    connection.commit()


def drop_schema(connection) -> None:
    """Drop every table this module owns. Intended for tests only."""

    with connection.cursor() as cursor:
        cursor.execute(DROP_SCHEMA_SQL)
    connection.commit()
