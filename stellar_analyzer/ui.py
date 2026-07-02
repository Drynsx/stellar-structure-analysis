"""Human-friendly terminal presentation for Stellar Analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

THEME = Theme({
    "accent": "#D97757",
    "accent_bold": "bold #D97757",
    "muted": "#8B8B86",
    "muted_bold": "bold #8B8B86",
    "good": "#4D9B76",
    "warn": "#D4A84B",
})
console = Console(theme=THEME)
error_console = Console(stderr=True, theme=THEME)


def banner() -> None:
    title = Text("Stellar", style="accent_bold")
    title.append("  structure analysis", style="muted")
    console.print(Panel(title, subtitle="MESA + physics-informed learning", border_style="accent", padding=(1, 2)))


def success(message: str) -> None:
    console.print(f"[good]OK[/good] {message}")


def show_profiles(snapshots: list[dict[str, Any]]) -> None:
    table = Table(title="MESA snapshots", title_style="accent_bold", header_style="muted_bold", box=None)
    table.add_column("Profile", justify="right")
    table.add_column("Model", justify="right")
    table.add_column("File", style="muted")
    for item in snapshots:
        table.add_row(str(item["profile_number"]), str(item["model_number"]), Path(item["path"]).name)
    console.print(table)
    console.print(f"[muted]{len(snapshots)} snapshots available[/muted]")


def _number(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.{digits}g}"


def show_analysis(result: dict[str, Any]) -> None:
    star = result["input"]
    fit = result["global_fit"]
    status = result["status"]
    status_style = "good" if status == "Normal" else "warn"
    heading = Text(star.get("name", "Stellar model"), style="bold")
    heading.append(f"   {status}", style=status_style)

    overview = Table.grid(padding=(0, 3))
    overview.add_column(style="muted")
    overview.add_column(justify="right")
    overview.add_row("Mass", f"{_number(star['mass'])} solar masses")
    overview.add_row("Temperature", f"{_number(star['teff'])} K")
    overview.add_row("Age", f"{_number(star['age'])} Gyr")
    overview.add_row("Global n", _number(fit["n_global"]))
    overview.add_row("Reduced chi-squared", _number(fit["reduced_chi2"]))
    overview.add_row("Anomaly score", _number(result["anomaly_score"]))
    console.print(Panel(overview, title=heading, border_style="accent", padding=(1, 2)))

    factors = Table(title="Physical contributions", title_style="accent_bold", header_style="muted_bold", box=None)
    factors.add_column("Effect")
    factors.add_column("Delta n", justify="right")
    labels = {"delta_n_rad": "Radiation", "delta_n_mu": "Composition", "delta_n_conv": "Convection",
              "delta_n_nuc": "Nuclear", "delta_n_deg": "Degeneracy"}
    for key, value in result["deviation_factors"].items():
        factors.add_row(labels.get(key, key), _number(value))
    console.print(factors)

    issues = result.get("preprocessing", {}).get("issues", [])
    if issues:
        console.print(Panel("\n".join(f"- {issue}" for issue in issues), title="Data notes", border_style="warn"))


def show_error(message: str) -> None:
    error_console.print(Panel(message, title="Could not complete that", border_style="red"))
