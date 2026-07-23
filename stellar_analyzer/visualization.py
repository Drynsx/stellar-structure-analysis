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
    baseline = 0.0 if float(np.nanmin(y)) < 0 < float(np.nanmax(y)) else float(np.nanmin(y))
    axis.fill_between(x, y, baseline, color="#2563EB", alpha=0.065)
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
    axis.text(1, -0.16, f"{len(x):,} valid radial samples", transform=axis.transAxes,
              ha="right", va="top", fontsize=8.5, color="#64748B")
    if field == "local-n":
        diagnostics = result.get("local_n_diagnostics", {})
        if diagnostics.get("warning"):
            axis.text(
                0.012,
                0.965,
                f"Local n quality: {diagnostics.get('status', 'warning')}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.7,
                color="#92400E",
                bbox={
                    "boxstyle": "round,pad=0.38",
                    "facecolor": "#FEF3C7",
                    "edgecolor": "#F59E0B",
                    "linewidth": 0.8,
                },
            )
    return figure


def _metric(parent, label: str, value: str) -> None:
    """Render a compact, readable stellar metadata card."""
    import tkinter as tk

    card = tk.Frame(parent, bg="#FFFFFF", padx=14, pady=8, highlightbackground="#E2E8F0",
                    highlightthickness=1)
    card.pack(side="left", padx=(0, 8))
    tk.Label(card, text=label.upper(), bg="#FFFFFF", fg="#64748B",
             font=("Segoe UI Semibold", 7)).pack(anchor="w")
    tk.Label(card, text=value, bg="#FFFFFF", fg="#0F172A",
             font=("Segoe UI Semibold", 10)).pack(anchor="w")


def _graph_toolbar(parent, canvas, navigation_toolbar) -> None:
    """Render an explicit graph control bar that remains visible at all sizes."""
    import tkinter as tk
    from tkinter import ttk

    bar = tk.Frame(parent, bg="#FFFFFF", padx=12, pady=8,
                   highlightbackground="#E2E8F0", highlightthickness=1)
    bar.pack(fill="x", padx=8, pady=(8, 0))
    tk.Label(bar, text="GRAPH CONTROLS", bg="#FFFFFF", fg="#64748B",
             font=("Segoe UI Semibold", 7)).pack(side="left", padx=(0, 12))
    actions = (
        ("Reset", navigation_toolbar.home),
        ("← Back", navigation_toolbar.back),
        ("Forward →", navigation_toolbar.forward),
        ("Pan", navigation_toolbar.pan),
        ("Zoom", navigation_toolbar.zoom),
        ("Save image…", navigation_toolbar.save_figure),
    )
    for text, command in actions:
        ttk.Button(bar, text=text, command=command, takefocus=True).pack(side="left", padx=(0, 6))
    tk.Label(bar, text="Ctrl+S: save  •  Ctrl+R: reset  •  Esc: close", bg="#FFFFFF", fg="#94A3B8",
             font=("Segoe UI", 8)).pack(side="right")
    canvas.get_tk_widget().bind("<Control-s>", lambda _event: navigation_toolbar.save_figure())
    canvas.get_tk_widget().bind("<Control-r>", lambda _event: navigation_toolbar.home())


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
    root.geometry("1120x760")
    root.minsize(820, 560)
    root.configure(bg="#F1F5F9")

    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    style.configure("Graph.TNotebook", background="#F1F5F9", borderwidth=0)
    style.configure("Graph.TNotebook.Tab", padding=(18, 9), font=("Segoe UI", 10))
    style.configure("TButton", padding=(10, 5), font=("Segoe UI", 9))
    style.map("Graph.TNotebook.Tab", foreground=[("selected", "#1D4ED8")],
              background=[("selected", "#FFFFFF")])

    header = tk.Frame(root, bg="#0F172A", padx=24, pady=16)
    header.pack(fill="x")
    title_group = tk.Frame(header, bg="#0F172A")
    title_group.pack(side="left")
    tk.Label(title_group, text="Stellar structure", bg="#0F172A", fg="#F8FAFC",
             font=("Segoe UI Semibold", 18)).pack(anchor="w")
    tk.Label(title_group, text="Interactive radial profile explorer", bg="#0F172A", fg="#94A3B8",
             font=("Segoe UI", 9)).pack(anchor="w")
    source = result.get("input", {}).get("name", "MESA profile")
    profile_number = result.get("source", {}).get("profile_number")
    detail = source + (f"  •  Profile {profile_number}" if profile_number is not None else "")
    tk.Label(header, text=detail, bg="#1E293B", fg="#CBD5E1", padx=12, pady=6,
             font=("Segoe UI Semibold", 9)).pack(side="right", pady=(5, 0))

    metadata = tk.Frame(root, bg="#F1F5F9", padx=22, pady=12)
    metadata.pack(fill="x")
    stellar = result.get("input", {})
    fit = result.get("global_fit", {})
    _metric(metadata, "Mass", f"{stellar.get('mass', '—'):.4g} M☉" if isinstance(stellar.get("mass"), (int, float)) else "—")
    _metric(metadata, "Temperature", f"{stellar.get('teff', '—'):.0f} K" if isinstance(stellar.get("teff"), (int, float)) else "—")
    _metric(metadata, "Age", f"{stellar.get('age', '—'):.4g} Gyr" if isinstance(stellar.get("age"), (int, float)) else "—")
    _metric(metadata, "Global n", f"{fit.get('n_global', '—'):.3f}" if isinstance(fit.get("n_global"), (int, float)) else "—")
    tk.Label(metadata, text="Use the toolbar to zoom, pan, reset, or export",
             bg="#F1F5F9", fg="#64748B", font=("Segoe UI", 9)).pack(side="right", pady=12)

    notebook = ttk.Notebook(root, style="Graph.TNotebook")
    notebook.pack(fill="both", expand=True, padx=14, pady=(0, 14))
    tabs: dict[str, tk.Frame] = {}
    for field, (_, title, _, _) in PLOT_FIELDS.items():
        tab = tk.Frame(notebook, bg="#F8FAFC")
        notebook.add(tab, text=title)
        figure = create_figure(result, field)
        canvas = FigureCanvasTkAgg(figure, master=tab)
        navigation_toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
        _graph_toolbar(tab, canvas, navigation_toolbar)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(8, 0))
        tabs[field] = tab
    notebook.select(tabs[initial_field])
    root.bind("<Escape>", lambda _event: root.destroy())
    root.mainloop()
