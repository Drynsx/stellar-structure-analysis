import json
from pathlib import Path

import numpy as np
import torch

from stellar_analyzer.cli import build_parser, main
from stellar_analyzer.ml.pinn_model import StellarPINN, build_differentiable_features, physics_residual_loss
from stellar_analyzer.ml.training_data import inspect_dataset


ROOT = Path(__file__).parents[1]


def test_cli_parses_mesa_analysis_and_training_commands():
    parser = build_parser()
    analysis = parser.parse_args(["analyze", "mesa", "--profile", "8"])
    training = parser.parse_args(["train-pinn", "--epochs", "2", "--device", "cpu"])
    assert analysis.profile == 8
    assert training.epochs == 2


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
    assert main(["plot", "density", "--profile", "8", "--width", "30", "--height", "8"]) == 0
    output = capsys.readouterr().out
    assert "Density" in output
    assert "radius / R" in output


def test_cli_saves_png_plot(tmp_path):
    output = tmp_path / "density.png"
    assert main(["plot", "density", "--profile", "8", "--save", str(output), "--save-only"]) == 0
    assert output.read_bytes().startswith(b"\x89PNG")


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
