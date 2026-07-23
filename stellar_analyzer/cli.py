"""Command-line interface for analysis and PINN lifecycle operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rich.panel import Panel

from stellar_analyzer.core.deviation_drivers import ANOMALY_THRESHOLD
from stellar_analyzer.core.data_loader import list_mesa_profiles
from stellar_analyzer.core.pipeline import analyze_mesa_job, analyze_profile, analyze_star, batch_analyze
from stellar_analyzer.ui import (
    banner, command_header, console, show_analysis, show_command_guide, show_error, show_profiles, show_workflow, success,
)
from stellar_analyzer.visualization import PLOT_FIELDS
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB = ROOT / "data" / "raw" / "MESA-Web_Job_03242664908"
DEFAULT_DATASET = ROOT / "data" / "processed" / "pinn_dataset.npz"
PROFILE_EXTENSIONS = {".data", ".profile", ".txt", ".csv"}


COMMAND_GUIDES = {
    "profiles": {
        "rows": [("--job", "MESA-Web job folder; leave blank for bundled example data"), ("--json", "print the snapshot list as JSON")],
        "examples": [".\\stellar profiles", ".\\stellar profiles --job data\\raw\\MESA-Web_1.0M"],
    },
    "analyze": {
        "rows": [("star", "type mass, effective temperature, metallicity, and age"), ("mesa", "choose a MESA-Web job and profile number"), ("profile", "give one uploaded MESA/MIST/BaSTI radial-profile file")],
        "examples": [".\\stellar analyze mesa --profile 8", ".\\stellar analyze star --mass 1 --teff 5778 --age 4.6", ".\\stellar analyze profile data\\uploads\\mist\\profile1.data"],
    },
    "screen": {
        "rows": [("catalog", "CSV upload with name,mass,teff,metallicity,age columns"), ("mesa", "screen all or selected snapshots in a MESA-Web job"), ("profile", "screen one or more uploaded profile files"), ("folder", "screen every supported profile file inside a folder")],
        "examples": [".\\stellar screen catalog stars.csv --output outputs\\anomaly_array.json", ".\\stellar screen folder data\\uploads\\mist --format csv --output outputs\\mist_anomalies.csv"],
    },
    "plot": {
        "rows": [("field", "density, pressure, temperature, or local-n"), ("--profile", "MESA profile number to plot"), ("--save", "optional PNG export path")],
        "examples": [".\\stellar plot density --profile 8", ".\\stellar plot local-n --profile 8 --save outputs\\local_n.png"],
    },
    "batch": {
        "rows": [("input", "CSV catalog with mass,teff,metallicity,age"), ("--output", "CSV file for full batch analysis results")],
        "examples": [".\\stellar batch stars.csv --output outputs\\catalog_results.csv"],
    },
    "uncertainty": {
        "rows": [("--profile", "MESA profile number"), ("--bootstrap", "number of resamples, usually 1000"), ("--seed", "integer seed for reproducible output")],
        "examples": [".\\stellar uncertainty --profile 8 --bootstrap 1000 --seed 42 --output outputs\\uncertainty.json"],
    },
    "prepare-pinn": {
        "rows": [("--job", "repeat once per MESA-Web star/track"), ("--min-samples", "fail unless enough profiles are found"), ("--grid", "optional HDF5 radial-profile grid")],
        "examples": [".\\stellar prepare-pinn --job data\\raw\\MESA-Web_0.8M --job data\\raw\\MESA-Web_1.0M --min-samples 150"],
    },
    "dataset-info": {
        "rows": [("data", "prepared .npz or .h5 dataset path; defaults to bundled PINN dataset")],
        "examples": [".\\stellar dataset-info", ".\\stellar dataset-info data\\processed\\pinn_grid.h5"],
    },
    "train-pinn": {
        "rows": [("--data", "prepared PINN dataset"), ("--config", "training JSON config"), ("--epochs", "optional override for quick smoke tests")],
        "examples": [".\\stellar train-pinn --config configs\\pinn_training.json", ".\\stellar train-pinn --epochs 2 --device cpu"],
    },
    "predict": {
        "rows": [("--checkpoint", "trained PINN weights"), ("--mass/--teff/--age", "stellar attributes to predict")],
        "examples": [".\\stellar predict --mass 1 --teff 5778 --age 4.6 --output outputs\\sun_pinn.json"],
    },
    "validate-pinn": {
        "rows": [("--checkpoint", "trained PINN weights"), ("--profile", "MESA profile to compare against")],
        "examples": [".\\stellar validate-pinn --profile 8"],
    },
}


def _ask(prompt: str, *, default: str | None = None, choices: tuple[str, ...] = ()) -> str:
    """Prompt until the user supplies a usable value."""
    suffix = f" [{default}]" if default is not None else ""
    if choices:
        suffix += f" ({'/'.join(choices)})"
    while True:
        console.print(f"[muted]{prompt}{suffix}[/muted]")
        value = input("> ").strip()
        value = value or default or ""
        if value and (not choices or value in choices):
            return value
        console.print(f"[warning]Enter {'one of ' + ', '.join(choices) if choices else 'a value'}.[/warning]")


def _run_guide(_args) -> None:
    """Build and run a common command through a step-by-step prompt."""
    command_header("guide", "Guided command builder")
    console.print(Panel("Answer each field, or press Enter to accept the suggested value.",
                        title="[label] Guided setup [/label]", title_align="left", border_style="grey35"))
    task = _ask("What would you like to do", default="analyze", choices=("analyze", "plot", "profiles"))
    if task == "profiles":
        argv = ["profiles"]
    elif task == "plot":
        field = _ask("Which quantity", default="density", choices=tuple(PLOT_FIELDS))
        profile = _ask("MESA profile number", default="8")
        argv = ["plot", field, "--profile", profile]
    else:
        source = _ask("What kind of input", default="mesa", choices=("mesa", "star", "profile"))
        argv = ["analyze", source]
        if source == "mesa":
            argv += ["--profile", _ask("MESA profile number", default="8")]
        elif source == "profile":
            argv += [_ask("Path to the profile file")]
        else:
            argv += [
                "--name", _ask("Star name", default="Custom Star"),
                "--mass", _ask("Mass in solar masses", default="1"),
                "--teff", _ask("Effective temperature in kelvin", default="5778"),
                "--metallicity", _ask("Metallicity [Fe/H]", default="0"),
                "--age", _ask("Age in billions of years", default="4.6"),
            ]
    console.print(f"\n[muted]Running:[/muted] [accent].\\stellar {' '.join(argv)}[/accent]\n")
    if main(argv) != 0:
        raise RuntimeError("The guided command did not complete successfully.")


def _run_command_guide(command: str) -> None:
    """Run a command-specific question guide and execute the built command."""

    command_header(command, "Guided command builder")
    console.print(Panel("Answer each field, or press Enter to accept the suggested value.",
                        title="[label] Guided setup [/label]", title_align="left", border_style="grey35"))
    if command == "profiles":
        argv = ["profiles", "--job", _ask("MESA-Web job folder", default=str(DEFAULT_JOB))]
    elif command == "analyze":
        source = _ask("What kind of input", default="mesa", choices=("mesa", "star", "profile"))
        argv = ["analyze", source]
        if source == "mesa":
            argv += ["--job", _ask("MESA-Web job folder", default=str(DEFAULT_JOB))]
            argv += ["--profile", _ask("Profile number", default="8")]
        elif source == "star":
            argv += [
                "--name", _ask("Star name", default="Custom Star"),
                "--mass", _ask("Mass in solar masses", default="1"),
                "--teff", _ask("Effective temperature in kelvin", default="5778"),
                "--metallicity", _ask("Metallicity [Fe/H]", default="0"),
                "--age", _ask("Age in billions of years", default="4.6"),
            ]
        else:
            argv += [_ask("Path to uploaded profile file")]
        output = _ask("Output JSON path; type skip to print normally", default="skip")
        if output.lower() != "skip":
            argv += ["--output", output]
    elif command == "screen":
        source = _ask("What are users uploading", default="folder", choices=("folder", "catalog", "mesa", "profile"))
        argv = ["screen", source]
        if source == "folder":
            argv += [_ask("Folder containing MIST/profile files", default="data\\uploads\\mist")]
        elif source == "catalog":
            argv += [_ask("CSV catalog path", default="stars.csv")]
        elif source == "mesa":
            argv += ["--job", _ask("MESA-Web job folder", default=str(DEFAULT_JOB))]
            profiles = _ask("Profile numbers, comma-separated; type all for every profile", default="all")
            if profiles.lower() != "all":
                for profile in [item.strip() for item in profiles.split(",") if item.strip()]:
                    argv += ["--profile", profile]
        else:
            files = _ask("Profile file paths, comma-separated")
            argv += [item.strip() for item in files.split(",") if item.strip()]
        destination = _ask("Where should the anomaly array go", default="window", choices=("window", "json", "csv", "terminal"))
        if destination == "terminal":
            fmt = _ask("Terminal format", default="json", choices=("json", "csv"))
            argv += ["--terminal", "--format", fmt]
        elif destination in {"json", "csv"}:
            output = _ask("Output path", default=f"outputs\\{source}_anomaly_array.{destination}")
            argv += ["--format", destination, "--output", output]
    elif command == "plot":
        field = _ask("Which graph", default="density", choices=tuple(PLOT_FIELDS))
        argv = ["plot", field, "--profile", _ask("MESA profile number", default="8")]
        save = _ask("PNG save path; type skip to only open the graph", default="skip")
        if save.lower() != "skip":
            argv += ["--save", save]
    elif command == "batch":
        argv = ["batch", _ask("CSV catalog path", default="stars.csv"), "--output", _ask("Output CSV path", default="outputs\\catalog_results.csv")]
    elif command == "uncertainty":
        argv = [
            "uncertainty",
            "--profile", _ask("MESA profile number", default="8"),
            "--bootstrap", _ask("Bootstrap resamples", default="1000"),
            "--seed", _ask("Random seed", default="42"),
            "--output", _ask("Output JSON path", default="outputs\\uncertainty.json"),
        ]
    elif command == "prepare-pinn":
        jobs = _ask("MESA-Web job folders, comma-separated", default=str(DEFAULT_JOB))
        argv = ["prepare-pinn"]
        for job in [item.strip() for item in jobs.split(",") if item.strip()]:
            argv += ["--job", job]
        argv += ["--min-samples", _ask("Minimum profiles required", default="3")]
    elif command == "dataset-info":
        argv = ["dataset-info", _ask("Dataset path", default=str(DEFAULT_DATASET))]
    elif command == "train-pinn":
        argv = ["train-pinn", "--config", _ask("Training config path", default="configs\\pinn_training.json")]
        epochs = _ask("Epoch override; type skip to use config", default="skip")
        if epochs.lower() != "skip":
            argv += ["--epochs", epochs]
    elif command == "predict":
        argv = [
            "predict",
            "--mass", _ask("Mass in solar masses", default="1"),
            "--teff", _ask("Effective temperature in kelvin", default="5778"),
            "--metallicity", _ask("Metallicity [Fe/H]", default="0"),
            "--age", _ask("Age in billions of years", default="4.6"),
            "--output", _ask("Output JSON path", default="outputs\\prediction.json"),
        ]
    elif command == "validate-pinn":
        argv = ["validate-pinn", "--profile", _ask("MESA profile number", default="8")]
    else:
        raise ValueError(f"No interactive guide is defined for {command}")

    console.print(f"\n[muted]Running:[/muted] [accent].\\stellar {' '.join(argv)}[/accent]\n")
    if main(argv) != 0:
        raise RuntimeError("The guided command did not complete successfully.")


def _run_help(args) -> None:
    command = args.topic or "screen"
    if command == "all":
        for name in COMMAND_GUIDES:
            guide = COMMAND_GUIDES[name]
            show_command_guide(name, guide["rows"], guide["examples"])
        return
    if command not in COMMAND_GUIDES:
        raise ValueError(f"Unknown guide topic: {command}. Use one of: {', '.join(COMMAND_GUIDES)}")
    guide = COMMAND_GUIDES[command]
    show_command_guide(command, guide["rows"], guide["examples"])


def _maybe_show_guide(args, command: str) -> bool:
    if getattr(args, "guide", False):
        _run_command_guide(command)
        return True
    return False


def _write_json(value: object, output: str | None = None) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    else:
        print(text)


def _summary(result: dict) -> dict:
    return {
        "input": result["input"],
        "source": result["source"],
        "preprocessing": result["preprocessing"],
        "global_fit": result["global_fit"],
        "piecewise_fit": result["piecewise_fit"],
        "deviation_factors": result["deviation_factors"],
        "anomaly_score": result["anomaly_score"],
        "status": result["status"],
    }


def _screening_reason(status: str, anomaly_score: float, convective_delta: float | None = None) -> str:
    if status == "Anomaly":
        return (
            f"|delta_global| = {abs(anomaly_score):.3f} exceeds the {ANOMALY_THRESHOLD:.1f} "
            "threshold, suggesting unresolved physics beyond the five drivers."
        )
    if convective_delta is not None and abs(convective_delta) > 0.25:
        return (
            f"Deviation is explained by convection (delta_n_conv = {convective_delta:.3f}); "
            "surface/convection structure is not an unresolved anomaly."
        )
    return (
        f"|delta_global| = {abs(anomaly_score):.3f} remains below the {ANOMALY_THRESHOLD:.1f} "
        "threshold after subtracting the five physical drivers."
    )


def _screening_record(star_id: str, result: dict) -> dict:
    factors = result.get("deviation_factors", {})
    return {
        "star_profile_id": star_id,
        "mass": result.get("input", {}).get("mass"),
        "age_gyr": result.get("input", {}).get("age"),
        "teff_k": result.get("input", {}).get("teff"),
        "n_global": result.get("global_fit", {}).get("n_global"),
        "delta_n_rad": factors.get("delta_n_rad"),
        "delta_n_mu": factors.get("delta_n_mu"),
        "delta_n_conv": factors.get("delta_n_conv"),
        "delta_n_nuc": factors.get("delta_n_nuc"),
        "delta_n_deg": factors.get("delta_n_deg"),
        "delta_global": result.get("anomaly_score"),
        "threshold": ANOMALY_THRESHOLD,
        "classification": result.get("status"),
        "diagnostic_reason": _screening_reason(
            result.get("status", "Normal"),
            float(result.get("anomaly_score", 0.0)),
            factors.get("delta_n_conv"),
        ),
    }


def _write_records(records: list[dict], output: Path | None, output_format: str) -> None:
    if output_format == "csv":
        frame = pd.DataFrame(records)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output, index=False)
            success(f"Wrote {len(records)} anomaly-screening rows -> {output}")
        else:
            print(frame.to_csv(index=False))
        return
    _write_json(records, str(output) if output else None)


def _run_profiles(args) -> None:
    if _maybe_show_guide(args, "profiles"):
        return
    snapshots = list_mesa_profiles(args.job)
    if args.json:
        _write_json(snapshots)
        return
    command_header("profiles", "MESA-Web snapshots")
    show_profiles(snapshots)


def _run_analyze(args) -> None:
    if _maybe_show_guide(args, "analyze"):
        return
    if args.source == "star":
        result = analyze_star({"name": args.name, "mass": args.mass, "teff": args.teff,
                               "metallicity": args.metallicity, "age": args.age})
    elif args.source == "mesa":
        result = analyze_mesa_job(args.job, args.profile)
    else:
        result = analyze_profile(args.path, n_points=args.points)
    selected = result if args.full else _summary(result)
    if args.output or args.json:
        _write_json(selected, args.output)
    else:
        command_header("analyze", result["source"].get("type", args.source))
        show_workflow(["Loaded stellar profile", "Validated and normalized 500 radial samples",
                       "Fitted global and piecewise polytropes", "Calculated physical deviations"])
        show_analysis(result)


def _run_plot(args) -> None:
    if _maybe_show_guide(args, "plot"):
        return
    from stellar_analyzer.visualization import save_plot, show_plot_window, terminal_plot

    if args.save_only and not args.save:
        raise ValueError("--save-only requires --save PATH")
    result = analyze_mesa_job(args.job, args.profile)
    if args.terminal and not args.save_only:
        terminal_plot(result, args.field, args.width, args.height)
    elif not args.save_only:
        show_plot_window(result, args.field)
    if args.save:
        save_plot(result, args.field, args.save)


def _run_batch(args) -> None:
    if _maybe_show_guide(args, "batch"):
        return
    command_header("batch", Path(args.input).name)
    frame = pd.read_csv(args.input)
    result = batch_analyze(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    success(f"Analyzed {len(result)} stars -> {output}")


def _run_screen(args) -> None:
    if _maybe_show_guide(args, "screen"):
        return
    records: list[dict] = []
    if args.source == "catalog":
        frame = pd.read_csv(args.input)
        required = {"mass", "teff", "age"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Catalog is missing columns: {', '.join(sorted(missing))}")
        for index, row in frame.iterrows():
            name = str(row.get("name", f"star_{index + 1}"))
            result = analyze_star({
                "name": name,
                "mass": float(row["mass"]),
                "teff": float(row["teff"]),
                "metallicity": float(row.get("metallicity", 0.0)),
                "age": float(row["age"]),
            })
            records.append(_screening_record(name, result))
    elif args.source == "mesa":
        snapshots = list_mesa_profiles(args.job)
        selected = [item for item in snapshots if args.profile is None or int(item["profile_number"]) in args.profile]
        for item in selected:
            profile_number = int(item["profile_number"])
            result = analyze_mesa_job(args.job, profile_number)
            records.append(_screening_record(f"profile_{profile_number}", result))
    else:
        paths = getattr(args, "path", [])
        if args.source == "folder":
            paths = sorted(
                item for item in args.folder.rglob("*")
                if item.is_file() and item.suffix.lower() in PROFILE_EXTENSIONS
            )
            if not paths:
                raise ValueError(f"No supported profile files found in {args.folder}")
        for path in paths:
            result = analyze_profile(path, n_points=args.points)
            records.append(_screening_record(path.stem, result))

    if not records:
        raise ValueError("No stars/profiles were available for anomaly screening")
    if args.output or args.terminal:
        _write_records(records, args.output, args.format)
    else:
        from stellar_analyzer.visualization import show_screen_window
        show_screen_window(records)


def _run_uncertainty(args) -> None:
    if _maybe_show_guide(args, "uncertainty"):
        return
    from stellar_analyzer.core.uncertainty import bootstrap_global_n

    result = analyze_mesa_job(args.job, args.profile)
    profile = result["profile"]
    bootstrap = bootstrap_global_n(
        np.asarray(profile["radius"]), np.asarray(profile["rho"]),
        float(result["input"]["teff"]), args.bootstrap, args.seed,
    )
    samples = bootstrap.pop("samples")
    if args.include_samples:
        bootstrap["samples"] = samples.tolist()
    bootstrap["profile"] = args.profile
    bootstrap["job"] = str(args.job)
    if args.output:
        command_header("uncertainty", f"profile {args.profile}")
        show_workflow([f"Completed {bootstrap['n_success']} / {bootstrap['requested_resamples']} resampled fits",
                       "Calculated the 95% confidence interval"])
    _write_json(bootstrap, str(args.output) if args.output else None)


def _run_prepare(args) -> None:
    if _maybe_show_guide(args, "prepare-pinn"):
        return
    from stellar_analyzer.ml.training_data import prepare_hdf5_grid_dataset, prepare_mesa_datasets

    if args.grid:
        output = args.output or ROOT / "data" / "processed" / "pinn_grid.h5"
        _write_json(prepare_hdf5_grid_dataset(args.grid, output, args.points, args.limit))
    else:
        jobs = args.job or [DEFAULT_JOB]
        _write_json(prepare_mesa_datasets(jobs, args.output or DEFAULT_DATASET, args.points, args.min_samples))


def _training_config(args):
    from dataclasses import fields
    from stellar_analyzer.ml.pinn_model import TrainingConfig

    values = {}
    if args.config:
        values.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    valid = {field.name for field in fields(TrainingConfig)}
    unknown = set(values).difference(valid)
    if unknown:
        raise ValueError(f"Unknown training configuration: {', '.join(sorted(unknown))}")
    for name in valid:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    return TrainingConfig(**values)


def _run_train(args) -> None:
    if _maybe_show_guide(args, "train-pinn"):
        return
    from stellar_analyzer.ml.pinn_model import load_training_dataset, train_pinn

    config = _training_config(args)
    _write_json(train_pinn(load_training_dataset(args.data), config=config))


def _run_dataset_info(args) -> None:
    if _maybe_show_guide(args, "dataset-info"):
        return
    from stellar_analyzer.ml.training_data import inspect_dataset

    _write_json(inspect_dataset(args.data))


def _run_predict(args) -> None:
    if _maybe_show_guide(args, "predict"):
        return
    import torch
    from stellar_analyzer.ml.pinn_model import (
        DELTA_NAMES,
        build_input_features,
        inverse_transform_delta_n,
        load_pinn_weights,
    )

    radius = np.linspace(0.0, 1.0, args.points, dtype=np.float32)
    shape = np.ones_like(radius)
    features = build_input_features(args.mass * shape, args.teff * shape, args.metallicity * shape,
                                    args.age * shape, radius).astype(np.float32)
    model = load_pinn_weights(args.checkpoint)
    with torch.no_grad():
        prediction = model(torch.from_numpy(features).unsqueeze(0))
    value = {
        "input": {"mass": args.mass, "teff": args.teff, "metallicity": args.metallicity, "age": args.age},
        "delta_n": dict(zip(DELTA_NAMES, inverse_transform_delta_n(prediction["delta_n"][0]).tolist())),
        "profile_schema": ["theta", "scaled_log_rho", "scaled_log_pressure"],
        "profiles": prediction["profiles"][0].tolist(),
    }
    _write_json(value, args.output)


def _run_validate(args) -> None:
    if _maybe_show_guide(args, "validate-pinn"):
        return
    import torch
    from stellar_analyzer.core.data_loader import load_mesa_web_job
    from stellar_analyzer.ml.pinn_model import DELTA_NAMES, inverse_transform_delta_n, load_pinn_weights
    from stellar_analyzer.ml.training_data import build_model_tensors

    model_data = load_mesa_web_job(args.job, args.profile)
    features, expected_profiles, _, expected_delta, _ = build_model_tensors(model_data, args.points)
    model = load_pinn_weights(args.checkpoint)
    with torch.no_grad():
        prediction = model(torch.from_numpy(features).unsqueeze(0))
    predicted_delta = inverse_transform_delta_n(prediction["delta_n"][0]).numpy()
    predicted_profiles = prediction["profiles"][0].numpy()
    value = {
        "checkpoint": str(args.checkpoint),
        "profile": args.profile,
        "delta_mae": float(np.mean(np.abs(predicted_delta - expected_delta))),
        "profile_rmse": {
            name: float(np.sqrt(np.mean((predicted_profiles[:, index] - expected_profiles[:, index]) ** 2)))
            for index, name in enumerate(("theta", "scaled_log_rho", "scaled_log_pressure"))
        },
        "deviations": {
            name: {"analytical": float(expected_delta[index]), "pinn": float(predicted_delta[index]),
                   "absolute_error": float(abs(predicted_delta[index] - expected_delta[index]))}
            for index, name in enumerate(DELTA_NAMES)
        },
    }
    _write_json(value, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stellar",
        description="Explore stellar structure, MESA profiles, and PINN models.",
        epilog="New here? Run '.\\stellar guide' for a step-by-step command builder.",
    )
    parser.add_argument("--version", action="version", version="stellar-analyzer 0.3.0")
    commands = parser.add_subparsers(dest="command", required=True)

    guide = commands.add_parser("guide", help="build a command interactively, one question at a time")
    guide.set_defaults(func=_run_guide)

    help_command = commands.add_parser("help", help="show a compact guide card for a command")
    help_command.add_argument("topic", nargs="?", choices=[*COMMAND_GUIDES.keys(), "all"], help="command to explain")
    help_command.set_defaults(func=_run_help)

    profiles = commands.add_parser("profiles", help="list snapshots in a MESA-Web job")
    profiles.add_argument("--job", type=Path, default=DEFAULT_JOB)
    profiles.add_argument("--json", action="store_true")
    profiles.add_argument("--guide", action="store_true", help="show a guided command card")
    profiles.set_defaults(func=_run_profiles)

    analyze = commands.add_parser("analyze", help="run the scientific analysis")
    sources = analyze.add_subparsers(dest="source", required=True)
    star = sources.add_parser("star", help="analyze stellar parameters")
    star.add_argument("--name", default="Custom Star")
    star.add_argument("--mass", type=float, required=True, metavar="SOLAR_MASSES", help="stellar mass; Sun = 1")
    star.add_argument("--teff", type=float, required=True, metavar="KELVIN", help="effective temperature; Sun ~ 5778")
    star.add_argument("--metallicity", type=float, default=0.0, metavar="FE_H", help="metallicity [Fe/H] (default: 0)")
    star.add_argument("--age", type=float, required=True, metavar="GYR", help="age in billions of years")
    mesa = sources.add_parser("mesa", help="analyze a MESA-Web snapshot")
    mesa.add_argument("--job", type=Path, default=DEFAULT_JOB)
    mesa.add_argument("--profile", type=int)
    profile = sources.add_parser("profile", help="analyze a MESA/MIST/BaSTI profile")
    profile.add_argument("path", type=Path)
    profile.add_argument("--points", type=int, default=500)
    for source in (star, mesa, profile):
        source.add_argument("--output")
        source.add_argument("--json", action="store_true", help="print machine-readable JSON")
        source.add_argument("--full", action="store_true", help="include radial arrays in JSON output")
        source.add_argument("--guide", action="store_true", help="show a guided command card")
        source.set_defaults(func=_run_analyze)

    plot = commands.add_parser("plot", help="open professional radial profile graphs in a desktop window")
    plot.add_argument("field", choices=list(PLOT_FIELDS))
    plot.add_argument("--job", type=Path, default=DEFAULT_JOB)
    plot.add_argument("--profile", type=int)
    plot.add_argument("--save", type=Path, help="also save a high-resolution PNG")
    plot.add_argument("--save-only", action="store_true", help="save the PNG without opening a graph window")
    plot.add_argument("--terminal", action="store_true", help="draw an ASCII graph instead of opening the desktop window")
    plot.add_argument("--width", type=int, default=68)
    plot.add_argument("--height", type=int, default=16)
    plot.add_argument("--guide", action="store_true", help="show a guided command card")
    plot.set_defaults(func=_run_plot)

    batch = commands.add_parser("batch", help="analyze a CSV catalog")
    batch.add_argument("input", type=Path)
    batch.add_argument("--output", required=True)
    batch.add_argument("--guide", action="store_true", help="show a guided command card")
    batch.set_defaults(func=_run_batch)

    screen = commands.add_parser("screen", help="analyze many stars/profiles and output an anomaly array")
    screen_sources = screen.add_subparsers(dest="source", required=True)
    screen_catalog = screen_sources.add_parser("catalog", help="screen a CSV catalog with mass, teff, age columns")
    screen_catalog.add_argument("input", type=Path)
    screen_mesa = screen_sources.add_parser("mesa", help="screen every profile in a MESA-Web job")
    screen_mesa.add_argument("--job", type=Path, default=DEFAULT_JOB)
    screen_mesa.add_argument("--profile", type=int, action="append", help="profile number; repeat to screen selected profiles")
    screen_profile = screen_sources.add_parser("profile", help="screen one or more MESA/MIST/BaSTI profile files")
    screen_profile.add_argument("path", type=Path, nargs="+")
    screen_profile.add_argument("--points", type=int, default=500)
    screen_folder = screen_sources.add_parser("folder", help="screen every supported profile file in a folder")
    screen_folder.add_argument("folder", type=Path)
    screen_folder.add_argument("--points", type=int, default=500)
    for screen_source in (screen_catalog, screen_mesa, screen_profile, screen_folder):
        screen_source.add_argument("--output", type=Path)
        screen_source.add_argument("--format", choices=("json", "csv"), default="json")
        screen_source.add_argument("--terminal", action="store_true", help="print the anomaly array instead of opening the desktop table")
        screen_source.add_argument("--guide", action="store_true", help="show a guided command card")
        screen_source.set_defaults(func=_run_screen)

    uncertainty = commands.add_parser("uncertainty", help="bootstrap uncertainty for a MESA global fit")
    uncertainty.add_argument("--job", type=Path, default=DEFAULT_JOB)
    uncertainty.add_argument("--profile", type=int, required=True)
    uncertainty.add_argument("--bootstrap", type=int, default=1000, metavar="RESAMPLES")
    uncertainty.add_argument("--seed", type=int, default=42)
    uncertainty.add_argument("--output", type=Path)
    uncertainty.add_argument("--include-samples", action="store_true")
    uncertainty.add_argument("--guide", action="store_true", help="show a guided command card")
    uncertainty.set_defaults(func=_run_uncertainty)

    prepare = commands.add_parser("prepare-pinn", help="build a PINN dataset from one or more MESA-Web jobs")
    prepare.add_argument(
        "--job",
        type=Path,
        action="append",
        default=None,
        help="MESA-Web job folder; repeat for multiple stars/tracks",
    )
    prepare.add_argument("--grid", type=Path, help="radial-profile HDF5 grid (overrides --job)")
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--points", type=int, default=500)
    prepare.add_argument("--limit", type=int, help="prepare only the first N HDF5 profiles")
    prepare.add_argument("--min-samples", type=int, default=3, help="fail unless at least this many profiles are found")
    prepare.add_argument("--guide", action="store_true", help="show a guided command card")
    prepare.set_defaults(func=_run_prepare)

    info = commands.add_parser("dataset-info", help="inspect a prepared PINN dataset")
    info.add_argument("data", type=Path, nargs="?", default=DEFAULT_DATASET)
    info.add_argument("--guide", action="store_true", help="show a guided command card")
    info.set_defaults(func=_run_dataset_info)

    train = commands.add_parser("train-pinn", help="train and evaluate a PINN checkpoint")
    train.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    train.add_argument("--config", type=Path)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int, dest="batch_size")
    train.add_argument("--learning-rate", type=float, dest="learning_rate")
    train.add_argument("--lambda-physics", type=float, dest="lambda_physics")
    train.add_argument("--patience", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"])
    train.add_argument("--output", dest="output_path")
    train.add_argument("--guide", action="store_true", help="show a guided command card")
    train.set_defaults(func=_run_train)

    predict = commands.add_parser("predict", help="run a trained PINN checkpoint")
    predict.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "pinn_checkpoint.pt")
    predict.add_argument("--mass", type=float, required=True)
    predict.add_argument("--teff", type=float, required=True)
    predict.add_argument("--metallicity", type=float, default=0.0)
    predict.add_argument("--age", type=float, required=True)
    predict.add_argument("--points", type=int, default=500)
    predict.add_argument("--output")
    predict.add_argument("--guide", action="store_true", help="show a guided command card")
    predict.set_defaults(func=_run_predict)

    validate = commands.add_parser("validate-pinn", help="compare a PINN checkpoint with analytical fits")
    validate.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "pinn_checkpoint.pt")
    validate.add_argument("--job", type=Path, default=DEFAULT_JOB)
    validate.add_argument("--profile", type=int, default=8)
    validate.add_argument("--points", type=int, default=500)
    validate.add_argument("--output")
    validate.add_argument("--guide", action="store_true", help="show a guided command card")
    validate.set_defaults(func=_run_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        effective_argv = sys.argv[1:] if argv is None else argv
        if not effective_argv:
            banner()
            console.print("[muted]Start with[/muted]  [accent].\\stellar profiles[/accent]  [muted]or[/muted]  [accent].\\stellar --help[/accent]")
            return 0
        if "--guide" in effective_argv:
            topic = next((item for item in effective_argv if item in COMMAND_GUIDES), "screen")
            _run_command_guide(topic)
            return 0
        args = parser.parse_args(effective_argv)
        args.func(args)
        return 0
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        show_error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
