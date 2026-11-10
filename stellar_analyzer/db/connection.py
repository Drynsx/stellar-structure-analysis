"""Guarded psycopg2 connection helper for the PostgreSQL storage module."""

from __future__ import annotations

import os

DEFAULT_DSN_ENV_VAR = "STELLAR_ANALYZER_DATABASE_URL"


def _require_psycopg2():
    try:
        import psycopg2  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency is optional.
        raise ImportError(
            "psycopg2 is required for the PostgreSQL storage module; install it with "
            "'pip install psycopg2-binary' (or the project's 'db' extra)"
        ) from exc
    return psycopg2


def connect(dsn: str | None = None):
    """Open a psycopg2 connection, reading the DSN from an env var by default.

    Raises ``ImportError`` with an actionable message if psycopg2 is not
    installed, and ``RuntimeError`` if no DSN was given and
    ``STELLAR_ANALYZER_DATABASE_URL`` is not set, so callers get a clear
    signal instead of an opaque connection failure.
    """

    psycopg2 = _require_psycopg2()
    dsn = dsn or os.environ.get(DEFAULT_DSN_ENV_VAR)
    if not dsn:
        raise RuntimeError(
            f"No PostgreSQL DSN given and {DEFAULT_DSN_ENV_VAR} is not set; "
            "pass dsn=... or export the connection string, e.g. "
            "'postgresql://user:password@localhost:5432/stellar_analyzer'"
        )
    return psycopg2.connect(dsn)


def is_available(dsn: str | None = None) -> bool:
    """Return whether a live PostgreSQL connection can be opened right now.

    Used by the test suite to skip integration/validation tests cleanly
    when no PostgreSQL instance is reachable, rather than failing them.
    """

    try:
        connection = connect(dsn)
    except Exception:
        return False
    connection.close()
    return True
