# Stellar Structure Analyzer

A computational astrophysics toolkit for stellar-structure analysis, anomaly
screening, and physics-informed machine-learning preparation using polytropic
indices.

The system analyzes stellar profiles from MESA-Web, uploaded MIST/MESA-style
radial profile files, and simple CSV catalogs. It fits global and local
polytropic structure, evaluates five physical deviation drivers, and screens
stars for unresolved anomalies using the master residual

\[
\delta_{global} = (n_{observed} - n_{base}) - \sum \langle \Delta n_i \rangle .
\]

A star is classified as anomalous only when

```text
|delta_global| > 5.0
```

Large convection or surface spikes are not automatically anomalies if the
physical drivers explain them.

## Current capabilities

- Global polytropic fitting with physical \(n\), \(\rho_c\), \(\alpha\), and \(K\).
- Local polytropic-index calculation with fallback diagnostics.
- Equality-constrained piecewise polytropic fitting.
- Five deviation drivers:
  - radiation pressure
  - composition gradient
  - convection
  - nuclear concentration
  - degeneracy
- Batch anomaly screening with JSON/CSV output.
- Desktop anomaly-array table using Tkinter.
- Desktop radial-profile graph window for density, pressure, temperature, and local \(n\).
- Question-style CLI guides for common workflows.
- Reproducible bootstrap uncertainty summaries.
- PINN dataset preparation and training scaffolding.
- Chapter 3 evidence package under `paper_report_data/`.

## Install and run

Recommended local setup on Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Recommended local setup on macOS/Linux:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

After installation, every user can run the installed console command:

```powershell
stellar-analyzer --help
```

If the console command is not on PATH, use the universal Python module form:

```powershell
python -m stellar_analyzer --help
```

On Windows, this repository also includes a shortcut launcher:

```powershell
.\stellar --help
```

For a user-level install outside this repository, install directly from GitHub:

```powershell
python -m pip install "git+https://github.com/Drynsx/stellar-structure-analysis.git"
stellar-analyzer --help
```

For isolated command-line installation with `pipx`:

```powershell
python -m pip install --user pipx
python -m pipx ensurepath
pipx install "git+https://github.com/Drynsx/stellar-structure-analysis.git"
stellar-analyzer --help
```

For a guided workflow:

```powershell
.\stellar guide
```

For a command-specific question guide:

```powershell
.\stellar screen --guide
.\stellar analyze --guide
.\stellar batch --guide
```

For quick reference cards:

```powershell
.\stellar help
.\stellar help screen
.\stellar help all
```

## Analyze stellar profiles

List bundled MESA-Web snapshots:

```powershell
.\stellar profiles
```

Analyze a MESA-Web snapshot:

```powershell
.\stellar analyze mesa --profile 8
```

Save machine-readable JSON:

```powershell
.\stellar analyze mesa --profile 8 --output outputs\profile8.json
```

Analyze a manually uploaded radial profile file:

```powershell
.\stellar analyze profile data\uploads\mist\profile1.data
```

Analyze stellar parameters directly:

```powershell
.\stellar analyze star --name Sun --mass 1 --teff 5778 --age 4.6
```

## Screen for anomaly candidates

The main user-facing workflow is:

```text
upload star/profile data -> analyze all entries -> output anomaly array
```

Users can upload MIST/MESA-style profile files by placing them in a folder such
as:

```text
data\uploads\mist\
```

Then run:

```powershell
.\stellar screen folder data\uploads\mist
```

Without `--output`, this opens a Tkinter desktop anomaly-array table. The table
shows the star/profile ID, mass, age, effective temperature, global \(n\),
\(\delta_{global}\), classification, and diagnostic reason. Rows classified as
anomalies are highlighted.

To save the anomaly array instead:

```powershell
.\stellar screen folder data\uploads\mist --output outputs\mist_anomaly_array.json
.\stellar screen folder data\uploads\mist --format csv --output outputs\mist_anomaly_array.csv
```

Screen bundled MESA-Web snapshots:

```powershell
.\stellar screen mesa
.\stellar screen mesa --profile 2 --profile 8
```

Screen a CSV catalog with columns `name`, `mass`, `teff`, `metallicity`, and
`age`:

```powershell
.\stellar screen catalog stars.csv --output outputs\anomaly_array.json
```

## Visualize radial structure

Open the desktop graph window:

```powershell
.\stellar plot density --profile 8
.\stellar plot local-n --profile 8
```

Save a graph as PNG:

```powershell
.\stellar plot temperature --profile 8 --save outputs\temperature.png
```

Use a terminal-only graph:

```powershell
.\stellar plot density --profile 8 --terminal
```

On Windows, double-click:

```text
stellar-graphs.pyw
```

to open the graph workspace without a terminal.

## Batch analysis

For a simple catalog, provide CSV columns `mass`, `teff`, `metallicity`, and
`age`:

```powershell
.\stellar batch stars.csv --output outputs\catalog_results.csv
```

Use `screen catalog` when the desired output is an anomaly-candidate array.

## Uncertainty analysis

Run a reproducible 1,000-resample bootstrap analysis:

```powershell
.\stellar uncertainty --profile 8 --bootstrap 1000 --seed 42 --output outputs\profile8_uncertainty.json
```

The result records the success count, success rate, mean, standard deviation,
95% confidence interval, and seed.

## Prepare and train the PINN

The bundled dataset is generated from one solar-mass MESA-Web track with eight
snapshots. It is suitable for validating the pipeline, not for claiming a
generalizable production PINN.

Prepare the bundled dataset:

```powershell
.\stellar prepare-pinn
.\stellar dataset-info
```

For the thesis training target, use at least eight MESA-Web tracks and at least
150 total profiles:

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

Train and validate:

```powershell
.\stellar train-pinn --config configs\pinn_training.json
.\stellar validate-pinn --profile 8
```

Prediction example:

```powershell
.\stellar predict --mass 1 --teff 5778 --age 4.6 --output outputs\sun_pinn.json
```

## MIST and MESA data note

This program requires internal radial structure profiles containing at least
radius, density, and pressure. Public MIST packaged grids are usually
evolutionary-track tables, not full internal radial profiles. If using MIST,
provide exported structure/profile files or a radial-profile HDF5 grid.

For a large HDF5 radial-profile grid:

```powershell
.\stellar prepare-pinn --grid data\raw\mist_grid\profiles.h5
.\stellar train-pinn --data data\processed\pinn_grid.h5 --config configs\pinn_training.json
```

See:

```text
data\raw\mist_grid\README.md
```

## Report evidence

Regenerate the Chapter 3 evidence package:

```powershell
.venv\Scripts\python.exe scripts\build_paper_report_data.py
```

Include additional MESA-Web tracks:

```powershell
.venv\Scripts\python.exe scripts\build_paper_report_data.py `
  --extra-job data\raw\MESA-Web_0.8M `
  --extra-job data\raw\MESA-Web_2.0M `
  --extra-job data\raw\MESA-Web_5.0M
```

Important files:

- `paper_report_data/METHOD_TRACEABILITY.csv`
- `paper_report_data/MANIFEST.csv`
- `paper_report_data/03_part3_global_comparison/section_4_3_anomaly_screening.md`
- `paper_report_data/03_part3_global_comparison/anomaly_screening.csv`

## Test

```powershell
.venv\Scripts\python.exe -m pytest -q
```

On some Windows systems, pytest temporary-directory cleanup can fail because of
locked folders. In that case, run with a fresh workspace-owned base temp:

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp-run
```

## Current evidence status

- Current bundled MESA evidence: one 1.0 \(M_\odot\) MESA-Web track with eight
  snapshots.
- Current anomaly screening output: all bundled snapshots are normal under the
  \(|\delta_{global}| > 5.0\) criterion.
- PINN architecture and dataset tooling are implemented.
- A scientifically generalizable PINN still requires the planned multitrack
  dataset, ideally at least eight stars/tracks and at least 150 profiles.
