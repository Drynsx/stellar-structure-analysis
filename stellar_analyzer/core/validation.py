"""External-dataset readiness and track-isolated validation helpers."""

from __future__ import annotations

from collections import Counter

import numpy as np


REQUIRED_MANIFEST_FIELDS = {
    "track_id", "source", "model_version", "mass_msun", "metallicity",
    "age_years", "evolution_stage", "sha256", "usage_rights",
}
TARGET_MASSES = (0.8, 1.0, 2.0, 5.0)


def leave_one_track_out(track_ids) -> list[dict[str, list[int] | str]]:
    """Return index-based folds that never mix snapshots from the held-out track."""

    ids = [str(value) for value in track_ids]
    unique = sorted(set(ids))
    if len(unique) < 2:
        return []
    return [
        {
            "held_out_track": held_out,
            "train_indices": [index for index, value in enumerate(ids) if value != held_out],
            "validation_indices": [index for index, value in enumerate(ids) if value == held_out],
        }
        for held_out in unique
    ]


def assess_multitrack_manifest(rows: list[dict]) -> dict:
    """Evaluate the Chapter 3 four-track, three-snapshot evidence gate."""

    if not rows:
        return {"ready": False, "errors": ["No external structure profiles are registered"]}
    missing_fields = sorted(REQUIRED_MANIFEST_FIELDS.difference(rows[0]))
    if missing_fields:
        return {"ready": False, "errors": [f"Missing manifest fields: {', '.join(missing_fields)}"]}

    counts = Counter(str(row["track_id"]) for row in rows)
    masses = [float(row["mass_msun"]) for row in rows]
    covered = [target for target in TARGET_MASSES if any(abs(value - target) <= max(0.1, 0.1 * target) for value in masses)]
    errors = []
    if len(counts) < 4:
        errors.append(f"Need at least four tracks; found {len(counts)}")
    undersampled = sorted(track for track, count in counts.items() if count < 3)
    if undersampled:
        errors.append(f"Tracks need at least three snapshots: {', '.join(undersampled)}")
    missing_masses = [target for target in TARGET_MASSES if target not in covered]
    if missing_masses:
        errors.append("Missing target mass coverage: " + ", ".join(map(str, missing_masses)))
    if any(not str(row.get("usage_rights", "")).strip() for row in rows):
        errors.append("Every source requires a usage-rights statement")

    return {
        "ready": not errors,
        "track_count": len(counts),
        "profile_count": len(rows),
        "snapshots_per_track": dict(sorted(counts.items())),
        "target_masses_msun": list(TARGET_MASSES),
        "covered_target_masses_msun": covered,
        "leave_one_track_out_folds": len(leave_one_track_out([row["track_id"] for row in rows])),
        "errors": errors,
    }
