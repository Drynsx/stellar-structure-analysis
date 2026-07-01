"""Compatibility entry point for preprocessing the bundled MESA-Web job."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stellar_analyzer.core.data_loader import load_mesa_web_job
from stellar_analyzer.core.preprocess import preprocess_profile


DEFAULT_JOB = PROJECT_ROOT / "data" / "raw" / "MESA-Web_Job_03242664908"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "mesa_profile_500_points.csv"


def process_mesa_profile(
    job_path: str | Path = DEFAULT_JOB,
    output_path: str | Path = DEFAULT_OUTPUT,
    profile_number: int | None = None,
) -> Path:
    """Load, normalize, and export one MESA-Web snapshot to 500 rows."""

    prepared = preprocess_profile(load_mesa_web_job(job_path, profile_number), n_points=500)
    profile = prepared.profile
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Normalized_Radius": profile["radius_fraction"],
            "Density": profile["rho"],
            "Pressure": profile["pressure"],
            "Temperature": profile.get("temperature"),
            "Mass_Enclosed": profile.get("mass_enclosed"),
        }
    ).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(process_mesa_profile())
