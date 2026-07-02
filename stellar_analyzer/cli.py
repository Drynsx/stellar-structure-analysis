"""Command-line interface for analysis and PINN lifecycle operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from stellar_analyzer.core.data_loader import list_mesa_profiles
from stellar_analyzer.core.pipeline import analyze_mesa_job, analyze_profile, analyze_star, batch_analyze
from stellar_analyzer.ui import banner, console, show_analysis, show_error, show_profiles, success
from stellar_analyzer.visualization import PLOT_FIELDS
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB = ROOT / "data" / "raw" / "MESA-Web_Job_03242664908"
DEFAULT_DATASET = ROOT / "data" / "processed" / "pinn_dataset.npz"


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


def _run_profiles(args) -> None:
    snapshots = list_mesa_profiles(args.job)
    if args.json:
        _write_json(snapshots)
        return
    show_profiles(snapshots)


def _run_analyze(args) -> None:
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
        show_analysis(result)


def _run_plot(args) -> None:
    from stellar_analyzer.visualization import save_plot, terminal_plot

    if args.save_only and not args.save:
        raise ValueError("--save-only requires --save PATH")
    result = analyze_mesa_job(args.job, args.profile)
    if not args.save_only:
        terminal_plot(result, args.field, args.width, args.height)
    if args.save:
        save_plot(result, args.field, args.save)


def _run_batch(args) -> None:
    frame = pd.read_csv(args.input)
    result = batch_analyze(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    success(f"Analyzed {len(result)} stars -> {output}")


def _run_prepare(args) -> None:
    from stellar_analyzer.ml.training_data import prepare_mesa_dataset

    _write_json(prepare_mesa_dataset(args.job, args.output, args.points))


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
    from stellar_analyzer.ml.pinn_model import NpzStellarDataset, train_pinn

    config = _training_config(args)
    _write_json(train_pinn(NpzStellarDataset(args.data), config=config))


def _run_dataset_info(args) -> None:
    from stellar_analyzer.ml.training_data import inspect_dataset

    _write_json(inspect_dataset(args.data))


def _run_predict(args) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stellar", description="Explore stellar structure, MESA profiles, and PINN models")
    parser.add_argument("--version", action="version", version="stellar-analyzer 0.3.0")
    commands = parser.add_subparsers(dest="command", required=True)

    profiles = commands.add_parser("profiles", help="list snapshots in a MESA-Web job")
    profiles.add_argument("--job", type=Path, default=DEFAULT_JOB)
    profiles.add_argument("--json", action="store_true")
    profiles.set_defaults(func=_run_profiles)

    analyze = commands.add_parser("analyze", help="run the scientific analysis")
    sources = analyze.add_subparsers(dest="source", required=True)
    star = sources.add_parser("star", help="analyze stellar parameters")
    star.add_argument("--name", default="Custom Star")
    star.add_argument("--mass", type=float, required=True)
    star.add_argument("--teff", type=float, required=True)
    star.add_argument("--metallicity", type=float, default=0.0)
    star.add_argument("--age", type=float, required=True)
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
        source.set_defaults(func=_run_analyze)

    plot = commands.add_parser("plot", help="display a radial profile graph")
    plot.add_argument("field", choices=list(PLOT_FIELDS))
    plot.add_argument("--job", type=Path, default=DEFAULT_JOB)
    plot.add_argument("--profile", type=int)
    plot.add_argument("--save", type=Path, help="also save a high-resolution PNG")
    plot.add_argument("--save-only", action="store_true", help="skip the terminal graph")
    plot.add_argument("--width", type=int, default=68)
    plot.add_argument("--height", type=int, default=16)
    plot.set_defaults(func=_run_plot)

    batch = commands.add_parser("batch", help="analyze a CSV catalog")
    batch.add_argument("input", type=Path)
    batch.add_argument("--output", required=True)
    batch.set_defaults(func=_run_batch)

    prepare = commands.add_parser("prepare-pinn", help="build a PINN dataset from a MESA-Web job")
    prepare.add_argument("--job", type=Path, default=DEFAULT_JOB)
    prepare.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument("--points", type=int, default=500)
    prepare.set_defaults(func=_run_prepare)

    info = commands.add_parser("dataset-info", help="inspect a prepared PINN dataset")
    info.add_argument("data", type=Path, nargs="?", default=DEFAULT_DATASET)
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
    train.set_defaults(func=_run_train)

    predict = commands.add_parser("predict", help="run a trained PINN checkpoint")
    predict.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "pinn_checkpoint.pt")
    predict.add_argument("--mass", type=float, required=True)
    predict.add_argument("--teff", type=float, required=True)
    predict.add_argument("--metallicity", type=float, default=0.0)
    predict.add_argument("--age", type=float, required=True)
    predict.add_argument("--points", type=int, default=500)
    predict.add_argument("--output")
    predict.set_defaults(func=_run_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        effective_argv = sys.argv[1:] if argv is None else argv
        if not effective_argv:
            banner()
            console.print("[muted]Start with[/muted]  [accent].\\stellar profiles[/accent]  [muted]or[/muted]  [accent].\\stellar --help[/accent]")
            return 0
        args = parser.parse_args(effective_argv)
        args.func(args)
        return 0
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        show_error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
