# Stellar Structure Analyzer CLI

Command-line tools for global, local, and piecewise polytropic analysis,
physical-deviation decomposition, legacy MESA-Web profiles, and reproducible
physics-informed neural network (PINN) training.

## Install

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

After installation, run commands through the repository launcher:

```powershell
.\stellar --help
```

## Analyze

List the bundled MESA snapshots:

```powershell
.\stellar profiles
```

Analyze the latest snapshot, with a concise JSON result:

```powershell
.\stellar analyze mesa
```

The default view is a readable terminal report. Add `--json` for machine-readable
output, or `--output` to write JSON to a file.

Display a radial graph directly in the terminal, or save a report-ready PNG:

```powershell
.\stellar plot density --profile 8
.\stellar plot temperature --profile 8 --save outputs\temperature.png
```

Available graphs are `density`, `pressure`, `temperature`, and `local-n`.

Analyze a particular snapshot and save full radial arrays:

```powershell
.\stellar analyze mesa --profile 8 --full --output outputs\profile8.json
```

Analyze stellar parameters or an external profile:

```powershell
.\stellar analyze star --name Sun --mass 1 --teff 5778 --age 4.6
.\stellar analyze profile path\to\profile.data
```

For a catalog, provide CSV columns `mass`, `teff`, `metallicity`, and `age`:

```powershell
.\stellar batch stars.csv --output outputs\catalog_results.csv
```

## Prepare and train the PINN

The bundled dataset is generated from all eight snapshots with no pickled
objects. Each sample contains 500 radial positions, 15 input features, three
profile targets, five physical-deviation targets, and a fitted polytropic index.
Wide-range deviation targets use a reversible signed-log transform during
optimization and are converted back to physical values for prediction output.

```powershell
.\stellar prepare-pinn
.\stellar dataset-info
.\stellar train-pinn --config configs\pinn_training.json
```

Training uses deterministic profile-level train/validation/test splits,
supervised profile and deviation losses, a Lane-Emden residual computed from
the model prediction, early stopping, and a metadata-rich checkpoint. A quick
pipeline check can use `--epochs 2 --output models\smoke_checkpoint.pt`.

After training:

```powershell
.\stellar predict --mass 1 --teff 5778 --age 4.6 --output outputs\sun_pinn.json
.\stellar validate-pinn --profile 8
```

For a large radial-profile HDF5 grid, preparation is streamed into a chunked
dataset and training reads it lazily:

```powershell
.\stellar prepare-pinn --grid data\raw\mist_grid\profiles.h5
.\stellar train-pinn --data data\processed\pinn_grid.h5 --config configs\pinn_training.json
```

The [official MIST packaged grids](https://mist.science/model_grids.html) are
evolutionary-track tables, not internal radial profiles. This model requires a
structure grid containing radius, density, and pressure for every sample; see
`data/raw/mist_grid/README.md`.

The bundled data contains one solar-mass evolutionary track. It is enough to
validate the training machinery, but not to claim a generalizable production
PINN. Add diverse MESA tracks across mass, metallicity, and evolutionary stage,
then rebuild the dataset before production training.

## Report evidence and tests

```powershell
.venv\Scripts\python.exe scripts\build_paper_report_data.py
.venv\Scripts\python.exe -m pytest -q
```

`paper_report_data/` maps the reproducible evidence to Chapter 3, PDF pages
18-42. `METHOD_TRACEABILITY.csv` locates each method and `MANIFEST.csv` records
file hashes.
