"""Streamlit frontend for the Stellar Polytropic Deviation Analyzer."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stellar_analyzer.core.data_loader import list_mesa_profiles
from stellar_analyzer.core.pipeline import analyze_mesa_job, analyze_profile, analyze_star, batch_analyze
from stellar_analyzer.utils.plotting import deviation_bar_figure, health_gauge_figure, profile_figure


BENCHMARKS = {
    "Sun": {"name": "Sun", "mass": 1.0, "teff": 5778.0, "metallicity": 0.0, "age": 4.6},
    "Procyon": {"name": "Procyon", "mass": 1.48, "teff": 6530.0, "metallicity": -0.05, "age": 1.9},
    "Sirius": {"name": "Sirius", "mass": 2.06, "teff": 9940.0, "metallicity": 0.4, "age": 0.24},
}
DEFAULT_MESA_JOB = PROJECT_ROOT / "data" / "raw" / "MESA-Web_Job_03242664908"


def _controls(defaults: dict) -> dict:
    cols = st.columns(4)
    mass = cols[0].number_input("Mass", min_value=0.08, max_value=80.0, value=float(defaults["mass"]), step=0.05)
    teff = cols[1].number_input("Teff", min_value=1500.0, max_value=60000.0, value=float(defaults["teff"]), step=50.0)
    metallicity = cols[2].number_input("Metallicity", min_value=-4.0, max_value=1.0, value=float(defaults["metallicity"]), step=0.05)
    age = cols[3].number_input("Age", min_value=0.0, max_value=14.0, value=float(defaults["age"]), step=0.1)
    return {"name": defaults.get("name", "Custom Star"), "mass": mass, "teff": teff, "metallicity": metallicity, "age": age}


def dashboard() -> None:
    st.title("Stellar Analyzer")
    source = st.segmented_control("Source", ["Synthetic", "MESA-Web Job", "Upload Profile"], default="MESA-Web Job")
    if source == "Synthetic":
        benchmark = st.selectbox("Benchmark", list(BENCHMARKS), index=0)
        params = _controls(BENCHMARKS[benchmark])
        params["name"] = st.text_input("Name", value=params["name"])
        if st.button("Analyze", type="primary"):
            st.session_state["result"] = analyze_star(params)
    elif source == "MESA-Web Job":
        snapshots = list_mesa_profiles(DEFAULT_MESA_JOB)
        labels = {
            f"Profile {item['profile_number']} · model {item['model_number']}": item["profile_number"]
            for item in snapshots
        }
        selected = st.selectbox("Snapshot", list(labels), index=max(len(labels) - 1, 0))
        if st.button("Analyze", type="primary"):
            st.session_state["result"] = analyze_mesa_job(str(DEFAULT_MESA_JOB), labels[selected])
    else:
        upload = st.file_uploader("MESA, MIST, or BaSTI profile", type=["data", "dat", "txt", "csv", "h5", "hdf5"])
        if upload is not None and st.button("Analyze", type="primary"):
            suffix = Path(upload.name).suffix or ".data"
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(upload.getvalue())
                    temp_path = handle.name
                st.session_state["result"] = analyze_profile(temp_path, params={"name": Path(upload.name).stem})
            finally:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)

    result = st.session_state.get("result")
    if not result:
        return

    tab_profiles, tab_decomp, tab_health = st.tabs(["Profiles", "Physics Decomposition", "Health Check"])
    with tab_profiles:
        st.plotly_chart(profile_figure(result), width="stretch")
    with tab_decomp:
        st.plotly_chart(deviation_bar_figure(result), width="stretch")
    with tab_health:
        cols = st.columns([1, 1])
        cols[0].plotly_chart(health_gauge_figure(result), width="stretch")
        factors = result["deviation_factors"]
        dominant = max(factors, key=lambda key: abs(factors[key]))
        cols[1].metric("Status", result["status"], f"{result['anomaly_score']:.3f}")
        cols[1].write(f"This star deviates most through {dominant.replace('_', ' ')}.")


def batch_page() -> None:
    st.title("Batch Upload")
    upload = st.file_uploader("CSV", type=["csv"])
    if upload is None:
        return
    frame = pd.read_csv(upload)
    with st.spinner("Processing"):
        enriched = batch_analyze(frame)
    st.dataframe(enriched.head(100), width="stretch")
    buffer = io.StringIO()
    enriched.to_csv(buffer, index=False)
    st.download_button("Download", buffer.getvalue(), file_name="stellar_batch_results.csv", mime="text/csv")


def settings_page() -> None:
    st.title("Settings")
    st.toggle("PINN Surrogate", value=True, key="use_pinn")
    st.toggle("Pre-computed Grid", value=False, key="use_grid")


def main() -> None:
    st.set_page_config(page_title="Stellar Analyzer", layout="wide")
    page = st.sidebar.radio("View", ["Dashboard", "Batch Upload", "Settings"])
    if page == "Dashboard":
        dashboard()
    elif page == "Batch Upload":
        batch_page()
    else:
        settings_page()


if __name__ == "__main__":
    main()
