"""Plotting helpers for API responses, reports, and Streamlit."""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stellar_analyzer.core.global_fit import _predict_density


def polytrope_density(profile: dict, global_fit: dict) -> np.ndarray:
    radius_fraction = np.asarray(profile["radius_fraction"], dtype=float)
    try:
        return _predict_density(
            radius_fraction,
            float(global_fit["n_global"]),
            float(global_fit["K"]),
            float(global_fit["alpha"]),
        )
    except Exception:
        return np.full_like(radius_fraction, np.nan)


def profile_figure(result: dict) -> go.Figure:
    profile = result["profile"]
    r = np.asarray(profile["radius_fraction"], dtype=float)
    rho = np.asarray(profile["rho"], dtype=float)
    pressure = np.asarray(profile["pressure"], dtype=float)
    temperature = np.asarray(profile["temperature"], dtype=float)
    rho_poly = polytrope_density(profile, result["global_fit"])

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Density", "Pressure", "Temperature"))
    fig.add_trace(go.Scatter(x=r, y=rho, name="Real rho", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=r, y=rho_poly, name="Polytrope rho", mode="lines", line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=r, y=pressure, name="Pressure", mode="lines"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r, y=temperature, name="Temperature", mode="lines"), row=1, col=3)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=3)
    fig.update_xaxes(title_text="r / R")
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h"))
    return fig


def deviation_bar_figure(result: dict) -> go.Figure:
    factors = result["deviation_factors"]
    labels = ["Radiation", "Composition", "Convection", "Nuclear", "Degeneracy"]
    values = np.asarray(
        [
            factors["delta_n_rad"],
            factors["delta_n_mu"],
            factors["delta_n_conv"],
            factors["delta_n_nuc"],
            factors["delta_n_deg"],
        ],
        dtype=float,
    )
    total = max(float(np.sum(np.abs(values))), 1e-99)
    percentages = np.abs(values) / total * 100.0
    fig = go.Figure(go.Bar(x=percentages, y=labels, orientation="h", marker_color=["#2f6f8f", "#a63d40", "#4f7d3a", "#d99b2b", "#6d5a99"]))
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=110, r=20, t=25, b=35),
        xaxis_title="Contribution (%)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def health_gauge_figure(result: dict) -> go.Figure:
    score = float(result["anomaly_score"])
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".3f"},
            gauge={
                "axis": {"range": [-0.25, 0.5]},
                "bar": {"color": "#a63d40" if score > 0.1 else "#2f7d59"},
                "steps": [
                    {"range": [-0.25, 0.1], "color": "#d9efe4"},
                    {"range": [0.1, 0.5], "color": "#f4d7d4"},
                ],
                "threshold": {"line": {"color": "#1f2937", "width": 3}, "value": 0.1},
            },
        )
    )
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=30, r=30, t=25, b=25))
    return fig


def plotly_json(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())
