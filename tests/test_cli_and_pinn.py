import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stellar_analyzer.cli import build_parser, main
from stellar_analyzer.core.local_fit import calculate_local_n_with_diagnostics
from stellar_analyzer.core.uncertainty import bootstrap_global_n, propagate_delta_n_rad_error
from stellar_analyzer.visualization import create_figure
from stellar_analyzer.ml.pinn_model import (
    Hdf5StellarDataset, StellarPINN, build_differentiable_features, physics_residual_loss,
)
from stellar_analyzer.ml.training_data import inspect_dataset, prepare_hdf5_grid_dataset, prepare_mesa_datasets


ROOT = Path(__file__).parents[1]


def test_bootstrap_is_deterministic_and_reports_validity():
    radius = np.linspace(1.0e8, 2.0e10, 24)
    rho = 100.0 * np.clip(1.0 - (radius / radius.max()) ** 2, 1e-5, None) ** 1.5
    first = bootstrap_global_n(radius, rho, 5778.0, n_bootstrap=5, random_state=7)
    second = bootstrap_global_n(radius, rho, 5778.0, n_bootstrap=5, random_state=7)
    assert first["requested_resamples"] == 5
    assert np.array_equal(first["samples"], second["samples"])
    assert first["success_rate"] == first["n_success"] / 5


def test_radiation_analytical_derivatives_match_finite_difference():
    result = propagate_delta_n_rad_error(0.8, 1.0e7, 100.0, 0.01, 1.0e4, 1.0)
    assert result["derivative_max_relative_error"] < 1e-6


def test_local_n_reports_full_fallback_instead_of_silent_flat_curve():
    radius = np.linspace(0.0, 1.0, 12)
    rho = np.exp(radius)
    pressure = rho.copy()
    n_local, diagnostics = calculate_local_n_with_diagnostics(pressure, rho, radius)
    assert np.allclose(n_local, 1.5)
    assert diagnostics.status == "fallback_all_1.5"
    assert diagnostics.fallback_fraction == 1.0
    assert diagnostics.warning is not None


def test_cli_parses_mesa_analysis_and_training_commands():
    parser = build_parser()
    analysis = parser.parse_args(["analyze", "mesa", "--profile", "8"])
    screen = parser.parse_args(["screen", "mesa", "--profile", "2", "--profile", "8", "--format", "csv"])
    screen_folder = parser.parse_args(["screen", "folder", "data/uploads/mist", "--format", "json"])
    training = parser.parse_args(["train-pinn", "--epochs", "2", "--device", "cpu"])
    prepare = parser.parse_args([
        "prepare-pinn", "--job", "data/raw/MESA-Web_0.8M", "--job", "data/raw/MESA-Web_1.0M",
        "--min-samples", "150",
    ])
    assert analysis.profile == 8
    assert screen.profile == [2, 8]
    assert screen.format == "csv"
    assert screen_folder.folder == Path("data/uploads/mist")
    assert training.epochs == 2
    assert len(prepare.job) == 2
    assert prepare.min_samples == 150
    validation = parser.parse_args(["validate-pinn", "--profile", "2"])
    assert validation.profile == 2


def test_cli_guide_builds_and_runs_a_command(capsys):
    with patch("builtins.input", side_effect=["profiles"]):
        assert main(["guide"]) == 0
    output = capsys.readouterr().out
    assert "Guided command builder" in output
    assert ".\\stellar profiles" in output
    assert "8 snapshots available" in output


def test_cli_command_guides_use_rich_ui(capsys, tmp_path):
    assert main(["help", "screen"]) == 0
    output = capsys.readouterr().out
    assert "stellar screen guide" in output
    assert "screen folder" in output
    catalog = tmp_path / "stars.csv"
    output_path = tmp_path / "catalog_results.csv"
    catalog.write_text("mass,teff,metallicity,age\n1,5778,0,4.6\n", encoding="utf-8")
    with patch("builtins.input", side_effect=[str(catalog), str(output_path)]):
        assert main(["batch", "--guide"]) == 0
    output = capsys.readouterr().out
    assert "Guided command builder" in output
    assert ".\\stellar batch" in output
    assert output_path.is_file()


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


def test_cli_screens_catalog_as_anomaly_array(tmp_path):
    catalog = tmp_path / "stars.csv"
    output = tmp_path / "anomalies.json"
    catalog.write_text(
        "name,mass,teff,metallicity,age\n"
        "solar_like,1.0,5778,0.0,4.6\n"
        "hot_star,5.0,15000,0.0,0.01\n",
        encoding="utf-8",
    )
    assert main(["screen", "catalog", str(catalog), "--output", str(output)]) == 0
    records = json.loads(output.read_text(encoding="utf-8"))
    assert [record["star_profile_id"] for record in records] == ["solar_like", "hot_star"]
    assert {"n_global", "delta_global", "classification", "diagnostic_reason"}.issubset(records[0])


def test_cli_screens_mesa_profiles_as_csv_array(tmp_path):
    output = tmp_path / "mesa_anomalies.csv"
    assert main([
        "screen", "mesa", "--profile", "2", "--profile", "8",
        "--format", "csv", "--output", str(output),
    ]) == 0
    frame = __import__("pandas").read_csv(output)
    assert frame["star_profile_id"].tolist() == ["profile_2", "profile_8"]
    assert frame["classification"].eq("Normal").all()


def test_cli_screens_folder_upload_as_anomaly_array(tmp_path):
    source = ROOT / "data" / "raw" / "MESA-Web_Job_03242664908" / "profile8.data"
    upload = tmp_path / "mist_upload"
    upload.mkdir()
    target = upload / "uploaded_profile.data"
    target.write_bytes(source.read_bytes())
    output = tmp_path / "folder_anomalies.json"
    assert main(["screen", "folder", str(upload), "--output", str(output)]) == 0
    records = json.loads(output.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["star_profile_id"] == "uploaded_profile"


def test_cli_opens_desktop_graph_window_by_default():
    with patch("stellar_analyzer.visualization.show_plot_window") as show_window:
        assert main(["plot", "local-n", "--profile", "8"]) == 0
    show_window.assert_called_once()
    assert show_window.call_args.args[1] == "local-n"


def test_professional_figure_has_clear_labels_and_sample_context():
    from stellar_analyzer.core.pipeline import analyze_mesa_job

    figure = create_figure(analyze_mesa_job(ROOT / "data" / "raw" / "MESA-Web_Job_03242664908", 8), "local-n")
    axis = figure.axes[0]
    assert axis.get_title(loc="left") == "Local polytropic index"
    assert axis.get_xlabel() == "Normalized radius  r / R"
    assert any("valid radial samples" in item.get_text() for item in axis.texts)


def test_local_n_figure_warns_when_data_is_fallback_filled():
    result = {
        "profile": {"radius_fraction": np.linspace(0.0, 1.0, 12)},
        "n_local": [1.5] * 12,
        "local_n_diagnostics": {
            "status": "fallback_all_1.5",
            "warning": "fallback",
        },
    }
    figure = create_figure(result, "local-n")
    axis = figure.axes[0]
    assert any("Local n quality: fallback_all_1.5" in item.get_text() for item in axis.texts)


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


def test_multi_job_mesa_dataset_records_track_metadata(tmp_path):
    job = ROOT / "data" / "raw" / "MESA-Web_Job_03242664908"
    output = tmp_path / "multi_job_dataset.npz"
    result = prepare_mesa_datasets([job, job], output, n_points=64, min_samples=16)
    metadata = inspect_dataset(output)
    assert result["samples"] == metadata["samples"] == 16
    assert metadata["track_count"] == 2
    assert metadata["features_shape"] == [16, 64, 15]
    with np.load(output, allow_pickle=False) as data:
        assert set(data["job_index"].tolist()) == {0, 1}


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
