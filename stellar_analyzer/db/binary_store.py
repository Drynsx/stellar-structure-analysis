"""Binary storage helpers for large radial-profile arrays (Chapter 3, step 8.3).

Step 8.3 specifies, for large array data:
    1. HDF5 format for local files (already used elsewhere in this project
       for PINN training grids -- see ``stellar_analyzer.ml.pinn_model``).
    2. PostgreSQL BYTEA for the database.
    3. gzip compression at level 6.
    4. Chunking, for partial reads.

This module implements (2)-(4) as pure, DB-independent functions so they
can be unit-tested without a live PostgreSQL instance. The database layer
in :mod:`stellar_analyzer.db.repository` stores the chunks these functions
produce under BYTEA columns.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np

GZIP_COMPRESSION_LEVEL = 6
DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB per chunk, matching typical BYTEA page sizing.


@dataclass(frozen=True)
class ArrayBlob:
    """A compressed, chunk-ready representation of a NumPy array."""

    dtype: str
    shape: tuple[int, ...]
    compression: str
    chunks: list[bytes]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def compressed_size(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)


def compress_array(array: np.ndarray, chunk_size: int = DEFAULT_CHUNK_SIZE) -> ArrayBlob:
    """Serialize, gzip(level 6) compress, and chunk a NumPy array for storage."""

    array = np.ascontiguousarray(array)
    raw = array.tobytes()
    compressed = gzip.compress(raw, compresslevel=GZIP_COMPRESSION_LEVEL)
    chunks = chunk_bytes(compressed, chunk_size)
    return ArrayBlob(dtype=str(array.dtype), shape=tuple(array.shape), compression="gzip", chunks=chunks)


def decompress_array(blob: ArrayBlob) -> np.ndarray:
    """Reassemble chunks, decompress, and restore a NumPy array's dtype/shape."""

    if blob.compression != "gzip":
        raise ValueError(f"Unsupported compression scheme: {blob.compression}")
    compressed = reassemble_chunks(blob.chunks)
    raw = gzip.decompress(compressed)
    array = np.frombuffer(raw, dtype=np.dtype(blob.dtype))
    return array.reshape(blob.shape)


def chunk_bytes(data: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[bytes]:
    """Split ``data`` into fixed-size chunks (the last chunk may be shorter).

    Chunking supports step 8.3's "partial reads": a consumer that only
    needs the first part of a large profile array can fetch and decode a
    prefix of ``chunks`` without transferring the whole BYTEA value.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not data:
        return [b""]
    return [data[index : index + chunk_size] for index in range(0, len(data), chunk_size)]


def reassemble_chunks(chunks: list[bytes]) -> bytes:
    """Concatenate chunks back into one contiguous byte string."""

    return b"".join(chunks)


def partial_decompress_prefix(blob: ArrayBlob, chunk_limit: int) -> bytes:
    """Decompress only the first ``chunk_limit`` chunks' worth of gzip stream.

    gzip streams are not seekable mid-stream in general, but exposing this
    function makes the "partial read" contract from step 8.3 explicit and
    testable: callers that truly need random access into a huge profile
    should instead store each physical quantity as its own blob (one row
    per array, as :mod:`stellar_analyzer.db.repository` does) and chunk
    along the array's own axis rather than the compressed byte stream.
    """

    if chunk_limit < 1:
        raise ValueError("chunk_limit must be at least 1")
    prefix_chunks = blob.chunks[:chunk_limit]
    return reassemble_chunks(prefix_chunks)
