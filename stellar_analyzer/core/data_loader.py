"""Load MIST, BaSTI, MESA-like, HDF5, and delimited stellar profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = {
    "radius": ("r", "radius", "rad", "radius_cm", "normalized_radius", "r_rsun", "r/R"),
    "radius_fraction": ("r_over_r", "r_fraction", "r_div_r", "x", "q_radius"),
    "rho": ("rho", "density", "dens", "logrho", "log_rho", "log10_rho"),
    "pressure": ("p", "pressure", "press", "logp", "log_p", "log10_p"),
    "temperature": ("t", "temp", "temperature", "logt", "log_t", "log10_t"),
    "mu": ("mu", "mean_molecular_weight", "mmw"),
    "epsilon": ("epsilon", "eps", "eps_nuc", "energy_generation", "nuclear_energy"),
    "grad_ad": ("grad_ad", "nabla_ad", "del_ad", "grada", "ad_grad"),
    "grad_rad": ("grad_rad", "nabla_rad", "del_rad", "gradr", "rad_grad"),
    "mass_enclosed": ("m", "mass", "mass_enclosed", "m_r", "mr", "q_mass"),
}


def _is_index_row(tokens: list[str]) -> bool:
    """Return whether tokens are MESA's numbered column guide row."""

    if len(tokens) < 2:
        return False
    try:
        numbers = [int(token) for token in tokens]
    except ValueError:
        return False
    return numbers == list(range(1, len(numbers) + 1))


def _mesa_sections(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = [line.split() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    guides = [index for index, tokens in enumerate(rows) if _is_index_row(tokens)]
    if not guides:
        raise ValueError(f"No MESA column guide found in {path}")

    table_guide = guides[-1]
    if table_guide + 2 >= len(rows):
        raise ValueError(f"MESA table is incomplete in {path}")
    columns = rows[table_guide + 1]
    data_rows = [row for row in rows[table_guide + 2 :] if len(row) == len(columns)]
    frame = pd.DataFrame(data_rows, columns=columns).apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all")

    metadata: dict[str, Any] = {"format": "mesa-web"}
    if len(guides) > 1:
        header_guide = guides[0]
        names = rows[header_guide + 1]
        values = rows[header_guide + 2]
        for name, value in zip(names, values):
            key = _normalize_name(name)
            try:
                metadata[key] = float(value)
            except ValueError:
                metadata[key] = value.strip('"')
    return metadata, frame


def _read_mesa_profile(path: Path) -> StellarModel:
    metadata, frame = _mesa_sections(path)
    arrays = _canonicalize_frame(frame)
    if not {"radius", "rho", "pressure"}.issubset(arrays):
        raise ValueError(f"MESA profile lacks radius, density, or pressure columns: {path}")

    metadata.update(
        {
            "radius_unit": "rsun",
            "mass_enclosed_unit": "msun",
            "radius_rsun": metadata.get("photosphere_r", float(np.nanmax(arrays["radius"]))),
            "mass_msun": metadata.get("star_mass", metadata.get("initial_mass", 1.0)),
            "source_profile": path.name,
        }
    )
    return StellarModel(arrays=arrays, metadata=metadata, source_path=str(path))


@dataclass
class StellarModel:
    """Canonical stellar profile container.

    Arrays are stored under names used by the computational modules:
    ``radius``, ``radius_fraction``, ``rho``, ``pressure``, ``temperature``,
    ``mu``, ``epsilon``, ``grad_ad``, ``grad_rad``, and ``mass_enclosed``.
    """

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    def require(self, *names: str) -> None:
        missing = [name for name in names if name not in self.arrays]
        if missing:
            raise ValueError(f"Missing required stellar profile arrays: {', '.join(missing)}")


def _normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _canonicalize_frame(df: pd.DataFrame) -> dict[str, np.ndarray]:
    normalized = {_normalize_name(col): col for col in df.columns}
    arrays: dict[str, np.ndarray] = {}

    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases:
            key = _normalize_name(alias)
            if key in normalized:
                values = pd.to_numeric(df[normalized[key]], errors="coerce").to_numpy(dtype=float)
                if canonical in {"rho", "pressure", "temperature"} and key.startswith("log"):
                    values = np.power(10.0, values)
                arrays[canonical] = values
                break

    if "radius" not in arrays and "radius_fraction" in arrays:
        arrays["radius"] = arrays["radius_fraction"]
    if "radius_fraction" not in arrays and "radius" in arrays:
        radius = np.asarray(arrays["radius"], dtype=float)
        max_radius = np.nanmax(np.abs(radius))
        if max_radius > 0:
            arrays["radius_fraction"] = radius / max_radius

    return arrays


def _read_ascii(path: Path) -> StellarModel:
    read_errors: list[str] = []
    for kwargs in (
        {"comment": "#", "sep": None, "engine": "python"},
        {"comment": "#", "sep": r"\s+", "engine": "python"},
        {"comment": "!", "sep": r"\s+", "engine": "python", "skiprows": 5},
    ):
        try:
            df = pd.read_csv(path, **kwargs)
            if len(df.columns) > 1 and not df.empty:
                arrays = _canonicalize_frame(df)
                if arrays:
                    return StellarModel(arrays=arrays, metadata={"format": "ascii"}, source_path=str(path))
        except Exception as exc:  # pragma: no cover - collected for a useful final error.
            read_errors.append(str(exc))

    raise ValueError(f"Could not parse stellar profile text file {path}: {' | '.join(read_errors)}")


def _read_hdf5(path: Path) -> StellarModel:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency is declared.
        raise ImportError("h5py is required to load HDF5 stellar models") from exc

    raw: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"format": "hdf5"}

    def visit(name: str, obj: Any) -> None:
        if hasattr(obj, "attrs"):
            for key, value in obj.attrs.items():
                if np.isscalar(value):
                    metadata[_normalize_name(key)] = value.item() if hasattr(value, "item") else value
        if hasattr(obj, "shape") and obj.shape is not None:
            data = np.asarray(obj)
            if data.ndim == 1 and np.issubdtype(data.dtype, np.number):
                raw[_normalize_name(Path(name).name)] = data.astype(float)

    with h5py.File(path, "r") as handle:
        handle.visititems(visit)
        for key, value in handle.attrs.items():
            metadata[_normalize_name(key)] = value.item() if hasattr(value, "item") else value

    df = pd.DataFrame(raw)
    arrays = _canonicalize_frame(df)
    if not arrays:
        raise ValueError(f"No recognizable 1-D stellar profile datasets found in {path}")
    return StellarModel(arrays=arrays, metadata=metadata, source_path=str(path))


def load_stellar_model(file_path: str | Path) -> StellarModel:
    """Read a stellar structure profile from HDF5 or an ASCII-like table.

    The loader accepts common MIST/BaSTI/MESA column spellings and returns a
    canonical :class:`StellarModel`. Logarithmic density, pressure, and
    temperature columns are converted back to linear values.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Stellar model file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5", ".hdf"}:
        return _read_hdf5(path)
    if suffix == ".data":
        try:
            return _read_mesa_profile(path)
        except ValueError:
            pass
    return _read_ascii(path)


def list_mesa_profiles(job_path: str | Path) -> list[dict[str, Any]]:
    """List snapshots in a MESA-Web job, ordered by model number."""

    job = Path(job_path)
    if not job.is_dir():
        raise FileNotFoundError(f"MESA-Web job directory not found: {job}")
    index_path = job / "profiles.index"
    snapshots: list[dict[str, Any]] = []
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 3 and all(part.lstrip("+-").isdigit() for part in parts[:3]):
                model_number, priority, profile_number = map(int, parts[:3])
                profile_path = job / f"profile{profile_number}.data"
                if profile_path.exists():
                    snapshots.append(
                        {
                            "model_number": model_number,
                            "priority": priority,
                            "profile_number": profile_number,
                            "path": str(profile_path),
                        }
                    )
    if not snapshots:
        for profile_path in sorted(job.glob("profile*.data")):
            digits = "".join(character for character in profile_path.stem if character.isdigit())
            snapshots.append(
                {
                    "model_number": int(digits or 0),
                    "priority": 0,
                    "profile_number": int(digits or 0),
                    "path": str(profile_path),
                }
            )
    return sorted(snapshots, key=lambda item: (item["model_number"], item["profile_number"]))


def load_mesa_web_job(job_path: str | Path, profile_number: int | None = None) -> StellarModel:
    """Load one snapshot from a legacy MESA-Web job (latest by default)."""

    snapshots = list_mesa_profiles(job_path)
    if not snapshots:
        raise ValueError(f"No profile snapshots found in MESA-Web job: {job_path}")
    if profile_number is None:
        selected = snapshots[-1]
    else:
        selected = next((item for item in snapshots if item["profile_number"] == profile_number), None)
        if selected is None:
            raise ValueError(f"Profile {profile_number} is not present in MESA-Web job: {job_path}")
    model = load_stellar_model(selected["path"])
    model.metadata.update(selected)
    model.metadata["job_path"] = str(Path(job_path))
    return model
