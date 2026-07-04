"""Desktop, terminal, and exported plots for analyzed stellar profiles."""

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
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = create_figure(result, field)
    figure.savefig(destination, dpi=180)
    import matplotlib.pyplot as plt
    plt.close(figure)
    success(f"Graph saved to {destination}")


def create_figure(result: dict[str, Any], field: str):
    """Create the publication-style figure shared by the GUI and PNG export."""
    from matplotlib.figure import Figure
    from matplotlib.ticker import AutoMinorLocator

    x, y, title, unit, _ = _series(result, field)
    if len(x) < 2:
        raise ValueError(f"Not enough valid values to plot {field}")

    figure = Figure(figsize=(9.6, 5.8), dpi=100, facecolor="#F8FAFC", constrained_layout=True)
    axis = figure.add_subplot(111)
    axis.set_facecolor("#FFFFFF")
    axis.plot(x, y, color="#2563EB", linewidth=2.35, solid_capstyle="round", label=title)
    axis.fill_between(x, y, float(np.nanmin(y)), color="#2563EB", alpha=0.07)
    axis.set_title(title, loc="left", fontsize=17, fontweight="semibold", color="#0F172A", pad=18)
    axis.text(0, 1.015, "Radial stellar structure profile", transform=axis.transAxes,
              fontsize=9.5, color="#64748B", va="bottom")
    axis.set_xlabel("Normalized radius  r / R", color="#334155", labelpad=10)
    axis.set_ylabel(unit, color="#334155", labelpad=10)
    axis.set_xlim(0, 1)
    axis.grid(True, which="major", color="#CBD5E1", linewidth=0.8, alpha=0.65)
    axis.grid(True, which="minor", color="#E2E8F0", linewidth=0.5, alpha=0.45)
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.tick_params(colors="#475569", labelsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#94A3B8")
    axis.legend(loc="best", frameon=False, fontsize=9)
    return figure


def show_plot_window(result: dict[str, Any], initial_field: str = "density") -> None:
    """Open a native desktop window with one professional tab per graph."""
    try:
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    except ImportError as exc:
        raise RuntimeError("The desktop graph window requires Tkinter and Matplotlib.") from exc

    root = tk.Tk()
    root.title("Stellar Analyzer — Radial Profiles")
    root.geometry("1100x720")
    root.minsize(820, 560)
    root.configure(bg="#F1F5F9")

    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    style.configure("Graph.TNotebook", background="#F1F5F9", borderwidth=0)
    style.configure("Graph.TNotebook.Tab", padding=(18, 9), font=("Segoe UI", 10))

    header = tk.Frame(root, bg="#0F172A", padx=24, pady=16)
    header.pack(fill="x")
    tk.Label(header, text="Stellar structure", bg="#0F172A", fg="#F8FAFC",
             font=("Segoe UI Semibold", 18)).pack(side="left")
    source = result.get("input", {}).get("name", "MESA profile")
    profile_number = result.get("source", {}).get("profile_number")
    detail = source + (f"  •  Profile {profile_number}" if profile_number is not None else "")
    tk.Label(header, text=detail, bg="#0F172A", fg="#94A3B8",
             font=("Segoe UI", 10)).pack(side="right", pady=(5, 0))

    notebook = ttk.Notebook(root, style="Graph.TNotebook")
    notebook.pack(fill="both", expand=True, padx=14, pady=14)
    tabs: dict[str, tk.Frame] = {}
    for field, (_, title, _, _) in PLOT_FIELDS.items():
        tab = tk.Frame(notebook, bg="#F8FAFC")
        notebook.add(tab, text=title)
        figure = create_figure(result, field)
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(8, 0))
        toolbar_frame = tk.Frame(tab, bg="#F8FAFC")
        toolbar_frame.pack(fill="x", padx=8, pady=(0, 8))
        NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False).pack(side="left")
        tabs[field] = tab
    notebook.select(tabs[initial_field])
    root.mainloop()
