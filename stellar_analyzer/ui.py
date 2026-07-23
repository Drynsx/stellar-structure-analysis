"""Human-friendly terminal presentation for Stellar Analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

THEME = Theme({
    "accent": "#35D07F",
    "accent_bold": "bold #35D07F",
    "muted": "#8A938E",
    "muted_bold": "bold #A7B0AB",
    "good": "#35D07F",
    "warn": "#E0A84B",
    "label": "bold black on white",
    "value": "#F2F4F3",
})
console = Console(theme=THEME)
error_console = Console(stderr=True, theme=THEME)


def banner() -> None:
    title = Text(" STELLAR ", style="label")
    title.append("  structure analysis", style="bold white")
    console.print(Panel(title, subtitle="MESA  |  polytropes  |  physics-informed learning",
                        subtitle_align="left", border_style="grey35", padding=(1, 2), box=box.ROUNDED))


def command_header(command: str, context: str = "") -> None:
    line = Text("stellar", style="muted")
    line.append(f"  {command}", style="bold white")
    if context:
        line.append(f"  {context}", style="muted")
    console.print(line)
    console.print()


def show_workflow(steps: list[str]) -> None:
    for step in steps:
        console.print(f"[good]>[/good] [value]{step}[/value]")
    console.print()


def success(message: str) -> None:
    console.print(f"[good]>[/good] {message}")


def show_command_guide(command: str, rows: list[tuple[str, str]], examples: list[str]) -> None:
    """Render a compact command guide using the same terminal styling."""

    table = Table(header_style="muted_bold", border_style="grey35", box=None,
                  padding=(0, 2), expand=True)
    table.add_column("Field")
    table.add_column("What to type")
    for field, description in rows:
        table.add_row(field, description)
    body = Table.grid(expand=True)
    body.add_row(table)
    if examples:
        body.add_row("")
        body.add_row(Text("Examples", style="muted_bold"))
        for example in examples:
            text = Text("> ", style="good")
            text.append(example, style="accent")
            body.add_row(text)
    console.print(Panel(body, title=f"[label] stellar {command} guide [/label]",
                        title_align="left", border_style="grey35", box=box.ROUNDED, padding=(1, 2)))


def show_profiles(snapshots: list[dict[str, Any]]) -> None:
    table = Table(header_style="muted_bold", border_style="grey35", box=None,
                  padding=(0, 2), expand=True)
    table.add_column("Profile", justify="right")
    table.add_column("Model", justify="right")
    table.add_column("File", style="muted")
    table.add_column("Status")
    for item in snapshots:
        table.add_row(str(item["profile_number"]), str(item["model_number"]), Path(item["path"]).name,
                      "[good]AVAILABLE[/good]")
    console.print(Panel(table, title="[label] MESA snapshots [/label]", title_align="left",
                        border_style="grey35", box=box.ROUNDED, padding=(0, 1)))
    console.print(f"[muted]  {len(snapshots)} snapshots available[/muted]")


def _number(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.{digits}g}"


def show_analysis(result: dict[str, Any]) -> None:
    star = result["input"]
    fit = result["global_fit"]
    status = result["status"]
    status_style = "good" if status == "Normal" else "warn"
    heading = Text(f" {star.get('name', 'Stellar model')} ", style="label")

    overview = Table.grid(padding=(0, 3))
    overview.add_column(style="muted", ratio=1)
    overview.add_column(justify="right", style="value", ratio=2)
    overview.add_row("State", f"[{status_style}]{status.upper()}[/{status_style}]")
    overview.add_row("Mass", f"{_number(star['mass'])} solar masses")
    overview.add_row("Temperature", f"{_number(star['teff'])} K")
    overview.add_row("Age", f"{_number(star['age'])} Gyr")
    overview.add_row("Global n", _number(fit["n_global"]))
    overview.add_row("Reduced chi-squared", _number(fit["reduced_chi2"]))
    overview.add_row("Anomaly score", _number(result["anomaly_score"]))
    console.print(Panel(overview, title=heading, title_align="left", border_style="grey35",
                        padding=(1, 2), box=box.ROUNDED))

    factors = Table(header_style="muted_bold", border_style="grey35", box=None,
                    padding=(0, 2), expand=True)
    factors.add_column("Effect")
    factors.add_column("Delta n", justify="right")
    labels = {"delta_n_rad": "Radiation", "delta_n_mu": "Composition", "delta_n_conv": "Convection",
              "delta_n_nuc": "Nuclear", "delta_n_deg": "Degeneracy"}
    for key, value in result["deviation_factors"].items():
        factors.add_row(labels.get(key, key), _number(value))
    console.print(Panel(factors, title="[label] Physical contributions [/label]", title_align="left",
                        border_style="grey35", box=box.ROUNDED, padding=(0, 1)))

    issues = result.get("preprocessing", {}).get("issues", [])
    if issues:
        console.print(Panel("\n".join(f"- {issue}" for issue in issues), title="[label] Data notes [/label]",
                            title_align="left", border_style="warn", box=box.ROUNDED))


def show_error(message: str) -> None:
    error_console.print(Panel(message, title="[bold white on red] ERROR [/bold white on red]",
                              title_align="left", border_style="red", box=box.ROUNDED))
