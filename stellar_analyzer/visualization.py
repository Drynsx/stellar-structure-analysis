"""Terminal and PNG plots for analyzed stellar profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stellar_analyzer.ui import console, success

PLOT_FIELDS = {
    "density": ("rho", "Density", "g cm^-3", True),
    "pressure": ("pressure", "Pressure", "dyn cm^-2", True),
    "temperature": ("temperature", "Temperature", "K", True),
    "local-n": ("n_local", "Local polytropic index", "n", False),
}


def _series(result: dict[str, Any], field: str) -> tuple[np.ndarray, np.ndarray, str, str, bool]:
    key, title, unit, log_scale = PLOT_FIELDS[field]
    profile = result["profile"]
    radius = np.asarray(profile["radius_fraction"], dtype=float)
    values = np.asarray(result["n_local"] if key == "n_local" else profile[key], dtype=float)
    mask = np.isfinite(radius) & np.isfinite(values)
    if log_scale:
        mask &= values > 0
        values = np.log10(np.clip(values, 1e-300, None))
        unit = f"log10({unit})"
    return radius[mask], values[mask], title, unit, log_scale


def terminal_plot(result: dict[str, Any], field: str, width: int = 68, height: int = 16) -> None:
    x, y, title, unit, _ = _series(result, field)
    if len(x) < 2:
        raise ValueError(f"Not enough valid values to plot {field}")
    width = max(30, min(width, 100))
    height = max(8, min(height, 30))
    grid = [[" " for _ in range(width)] for _ in range(height)]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xr, yr = max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12)
    columns = np.clip(((x - xmin) / xr * (width - 1)).astype(int), 0, width - 1)
    rows = np.clip(((ymax - y) / yr * (height - 1)).astype(int), 0, height - 1)
    for row, column in zip(rows, columns):
        grid[row][column] = "*"
    console.print(f"\n[accent_bold]{title}[/accent_bold]  [muted]{unit}[/muted]")
    console.print(f"[muted]{ymax:>10.3g} +[/muted]" + "".join(grid[0]))
    for row in grid[1:-1]:
        console.print("[muted]           |[/muted]" + "".join(row))
    console.print(f"[muted]{ymin:>10.3g} +[/muted]" + "".join(grid[-1]))
    console.print("[muted]           +" + "-" * width + " radius / R[/muted]")


def save_plot(result: dict[str, Any], field: str, destination: Path) -> None:
    import matplotlib.pyplot as plt

    x, y, title, unit, _ = _series(result, field)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.plot(x, y, color="#D97757", linewidth=2.2)
    axis.set(title=title, xlabel="Normalized radius (r/R)", ylabel=unit)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    success(f"Graph saved to {destination}")
