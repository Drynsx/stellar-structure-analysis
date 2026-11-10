"""PostgreSQL storage module (Chapter 3, step 8: database and storage).

Submodules:

* :mod:`stellar_analyzer.db.binary_store` -- gzip(6) compression and
  chunking for large radial-profile arrays (step 8.3). Pure Python/NumPy;
  no database connection required, so these are unit-testable on their own.
* :mod:`stellar_analyzer.db.schema` -- the PostgreSQL schema (step 8.1) and
  its indexes (step 8.2) for the three data levels defined in step 1
  (global parameters, radial profiles, derived quantities), plus the
  BYTEA-backed binary blob table (step 8.3).
* :mod:`stellar_analyzer.db.connection` -- a guarded psycopg2 connection
  helper that raises a clear ImportError when psycopg2 is not installed.
* :mod:`stellar_analyzer.db.repository` -- insert/query functions built on
  the schema above.
"""
