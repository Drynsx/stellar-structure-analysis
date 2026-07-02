# Stellar Polytropic Deviation Analyzer

Production-oriented analysis of stellar structure profiles with global, local,
and piecewise polytropic fits; five physical deviation drivers; FastAPI and
Streamlit interfaces; and a PyTorch PINN scaffold.

## Setup

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using `python -m pip` is supported everywhere. The direct
`.venv\Scripts\pip.exe` launcher is also repaired and usable.

## MESA-Web data

The legacy job is stored at:

```text
data/raw/MESA-Web_Job_03242664908/
```

Its eight snapshots are read from `profiles.index`. MESA radius and enclosed
mass units are converted from solar units to CGS, logarithmic density and
temperature are expanded, the surface-to-center rows are reordered, and every
profile is resampled onto a 500-point radial grid.

Analyze the latest snapshot:

```python
from stellar_analyzer import analyze_mesa_job

result = analyze_mesa_job("data/raw/MESA-Web_Job_03242664908")
print(result["source"], result["global_fit"]["n_global"])
```

Export a legacy-compatible processed CSV:

```powershell
.venv\Scripts\python.exe src\data_prep_mesa.py
```

## Run

```powershell
.venv\Scripts\uvicorn.exe stellar_analyzer.web.api:app --reload
.venv\Scripts\streamlit.exe run stellar_analyzer/web/streamlit_app.py
```

The dashboard can analyze synthetic benchmarks, any snapshot in the bundled
MESA-Web job, or an uploaded MESA/MIST/BaSTI profile.

## API

- `POST /analyze`: analyze stellar parameters with the surrogate profile.
- `GET /mesa/profiles`: list bundled MESA snapshots.
- `POST /mesa/analyze/{profile_number}`: analyze a bundled snapshot.
- `POST /mesa/upload`: analyze an uploaded structure profile.
- `POST /batch_analyze`: analyze a CSV catalog.
- `GET /export/{star_id}`: create a PDF report.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Research-paper evidence

Reproducible data for the report is stored in `paper_report_data/`. Its numbered
folders follow all four parts of Chapter 3 on PDF pages 18-42 of the provided
paper reference.

Regenerate the complete package from the raw MESA-Web job:

```powershell
.venv\Scripts\python.exe scripts\build_paper_report_data.py
```

Use `paper_report_data/METHOD_TRACEABILITY.csv` to locate the table supporting
each method and `paper_report_data/MANIFEST.csv` to verify file integrity. The
package explicitly records that the PINN has not yet been trained and that the
current evidence covers eight snapshots from one stellar track.
