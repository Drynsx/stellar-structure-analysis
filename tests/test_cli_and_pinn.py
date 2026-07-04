import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stellar_analyzer.cli import build_parser, main
from stellar_analyzer.ml.pinn_model import (
    Hdf5StellarDataset, StellarPINN, build_differentiable_features, physics_residual_loss,
)
from stellar_analyzer.ml.training_data import inspect_dataset, prepare_hdf5_grid_dataset


ROOT = Path(__file__).parents[1]


def test_cli_parses_mesa_analysis_and_training_commands():
    parser = build_parser()
    analysis = parser.parse_args(["analyze", "mesa", "--profile", "8"])
    training = parser.parse_args(["train-pinn", "--epochs", "2", "--device", "cpu"])
    assert analysis.profile == 8
    assert training.epochs == 2
    validation = parser.parse_args(["validate-pinn", "--profile", "2"])
    assert validation.profile == 2


def test_cli_guide_builds_and_runs_a_command(capsys):
    with patch("builtins.input", side_effect=["profiles"]):
        assert main(["guide"]) == 0
    output = capsys.readouterr().out
    assert "Guided command builder" in output
    assert ".\\stellar profiles" in output
    assert "8 snapshots available" in output


def test_cli_analyzes_mesa_to_json(tmp_path):
    output = tmp_path / "analysis.json"
    code = main(["analyze", "mesa", "--profile", "8", "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert result["source"]["profile_number"] == 8
    assert "profile" not in result


def test_cli_renders_analysis_and_terminal_plot(capsys):
    assert main(["analyze", "star", "--mass", "1", "--teff", "5778", "--age", "4.6"]) == 0
    assert "Physical contributions" in capsys.readouterr().out
    assert main(["plot", "density", "--profile", "8", "--terminal", "--width", "30", "--height", "8"]) == 0
    output = capsys.readouterr().out
    assert "Density" in output
    assert "radius / R" in output


def test_cli_saves_png_plot(tmp_path):
    output = tmp_path / "density.png"
    assert main(["plot", "density", "--profile", "8", "--save", str(output), "--save-only"]) == 0
    assert output.read_bytes().startswith(b"\x89PNG")


def test_cli_opens_desktop_graph_window_by_default():
    with patch("stellar_analyzer.visualization.show_plot_window") as show_window:
        assert main(["plot", "local-n", "--profile", "8"]) == 0
    show_window.assert_called_once()
    assert show_window.call_args.args[1] == "local-n"


def test_pinn_forward_and_physics_loss_are_differentiable():
    model = StellarPINN(width=16, depth=2)
    stored = torch.rand(2, 20, 15)
    stored[..., 4] = torch.linspace(0.0, 1.0, 20)
    features, radius = build_differentiable_features(stored)
    prediction = model(features)
    loss = physics_residual_loss(radius, prediction["profiles"][..., 0], torch.tensor([1.5, 3.0]))
    loss.backward()
    assert prediction["delta_n"].shape == (2, 5)
    assert prediction["profiles"].shape == (2, 20, 3)
    assert torch.isfinite(loss)


def test_prepared_dataset_has_expected_schema():
    path = ROOT / "data" / "processed" / "pinn_dataset.npz"
    metadata = inspect_dataset(path)
    assert metadata["features_shape"] == [8, 500, 15]
    assert metadata["profiles_shape"] == [8, 500, 3]
    assert metadata["delta_shape"] == [8, 5]
    with np.load(path, allow_pickle=False) as data:
        assert np.isfinite(data["features"]).all()
        assert np.isfinite(data["profiles"]).all()
        assert np.isfinite(data["delta_n"]).all()
        assert "delta_n_raw" in data


def test_hdf5_profile_grid_is_prepared_and_read_lazily(tmp_path):
    import h5py

    source = tmp_path / "profiles.h5"
    radius = np.linspace(0.001, 1.0, 20)
    with h5py.File(source, "w") as handle:
        for index in range(3):
            group = handle.create_group(f"model_{index}")
            group.attrs.update({"mass": 1.0, "teff": 5778.0, "metallicity": 0.0, "age": 4.6})
            group["radius"] = radius
            group["rho"] = 100.0 * (1.01 - radius) ** 2 + 1e-5
            group["pressure"] = 1e17 * (1.01 - radius) ** 3 + 1e8
            group["temperature"] = 1e7 * (1.01 - radius) + 5778.0
    output = tmp_path / "training.h5"
    result = prepare_hdf5_grid_dataset(source, output, n_points=32)
    dataset = Hdf5StellarDataset(output)
    assert result["samples"] == len(dataset) == 3
    assert dataset[0]["features"].shape == (32, 15)
    assert inspect_dataset(output)["features_shape"] == [3, 32, 15]
