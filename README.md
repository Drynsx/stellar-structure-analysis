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

Prefer to be walked through it? The interactive guide asks what you want to do,
offers sensible defaults, shows the command it builds, and runs it for you:

```powershell
.\stellar guide
```

At each prompt, press Enter to accept the value in square brackets. Regular
commands also include clearer value names and units in their `--help` output;
for example, run `.\stellar analyze star --help`.

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

Open a separate desktop graph window with professional tabs for density,
pressure, temperature, and local polytropic index. The window includes standard
zoom, pan, reset, and image-export controls:

```powershell
.\stellar plot density --profile 8
.\stellar plot temperature --profile 8 --save outputs\temperature.png
```

On Windows, double-click `stellar-graphs.pyw` to open the complete graph
workspace directly without a terminal window.

For a text-only session, add `--terminal` to use the compact ASCII graph.

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

To use the system as an anomaly screener, provide a catalog or a set of MESA
profiles and write the result as an array. Each row includes the fitted global
index, the five deviation drivers, `delta_global`, the `Normal`/`Anomaly`
classification, and a diagnostic reason. A star is flagged only when
`|delta_global| > 5.0`.

```powershell
.\stellar screen catalog stars.csv --output outputs\anomaly_array.json
.\stellar screen catalog stars.csv --format csv --output outputs\anomaly_array.csv
.\stellar screen mesa --output outputs\mesa_anomaly_array.json
.\stellar screen mesa --profile 2 --profile 8 --format csv --output outputs\selected_profiles.csv
.\stellar screen folder data\uploads\mist --format csv --output outputs\mist_anomaly_array.csv
```

Every command also has a question-style guide using the same terminal UI. It
asks what to type next, builds the command, shows it, and runs it:

```powershell
.\stellar screen --guide
.\stellar analyze --guide
.\stellar batch --guide
```

For a quick non-interactive reference card, use:

```powershell
.\stellar help screen
.\stellar help all
```

Run a reproducible 1,000-resample uncertainty analysis for a MESA snapshot:

```powershell
.\stellar uncertainty --profile 8 --bootstrap 1000 --seed 42 --output outputs\profile8-uncertainty.json
```

The summary records the successful-fit rate and 95% confidence interval. Add
`--include-samples` only when individual bootstrap values are required.

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

For the 150-profile training requirement, use at least eight MESA-Web stellar
tracks and fail fast if fewer than 150 profiles are present:

```powershell
.\stellar prepare-pinn `
  --job data\raw\MESA-Web_0.8M `
  --job data\raw\MESA-Web_1.0M `
  --job data\raw\MESA-Web_1.2M `
  --job data\raw\MESA-Web_1.5M `
  --job data\raw\MESA-Web_2.0M `
  --job data\raw\MESA-Web_3.0M `
  --job data\raw\MESA-Web_4.0M `
  --job data\raw\MESA-Web_5.0M `
  --min-samples 150
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

Additional genuine MESA-Web track directories can be included without mixing
their validation folds:

```powershell
.venv\Scripts\python.exe scripts\build_paper_report_data.py `
  --extra-job data\raw\MESA-Web_0.8M `
  --extra-job data\raw\MESA-Web_2M `
  --extra-job data\raw\MESA-Web_5M
```

`paper_report_data/` maps the reproducible evidence to Chapter 3, PDF pages
18-42. `METHOD_TRACEABILITY.csv` locates each method and `MANIFEST.csv` records
file hashes.
