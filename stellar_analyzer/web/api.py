"""FastAPI application for stellar polytropic deviation analysis."""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from stellar_analyzer.core.data_loader import list_mesa_profiles
from stellar_analyzer.core.pipeline import analyze_mesa_job, analyze_profile, analyze_star, batch_analyze
from stellar_analyzer.utils.plotting import deviation_bar_figure, health_gauge_figure, plotly_json, profile_figure

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Stellar Polytropic Deviation Analyzer", version="0.1.0")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MESA_JOB = PROJECT_ROOT / "data" / "raw" / "MESA-Web_Job_03242664908"


class AnalyzeRequest(BaseModel):
    mass: float = Field(1.0, ge=0.08, le=80.0)
    teff: float = Field(5778.0, ge=1500.0, le=60000.0)
    metallicity: float = Field(0.0, ge=-4.0, le=1.0)
    age: float = Field(4.6, ge=0.0, le=14.0)
    name: str = "Custom Star"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict:
    try:
        logger.info("Analyzing star %s", payload.name)
        payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        result = analyze_star(payload_dict)
        result["plots"] = {
            "profiles": plotly_json(profile_figure(result)),
            "decomposition": plotly_json(deviation_bar_figure(result)),
            "health": plotly_json(health_gauge_figure(result)),
        }
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail="Analysis failed") from exc


@app.get("/mesa/profiles")
def mesa_profiles() -> list[dict]:
    """Return snapshots available in the bundled legacy MESA-Web job."""

    try:
        return list_mesa_profiles(DEFAULT_MESA_JOB)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/mesa/analyze/{profile_number}")
def analyze_mesa_snapshot(profile_number: int) -> dict:
    try:
        return analyze_mesa_job(str(DEFAULT_MESA_JOB), profile_number=profile_number)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/mesa/upload")
async def analyze_mesa_upload(file: Annotated[UploadFile, File(...)]) -> dict:
    """Analyze an uploaded MESA profile without retaining the upload."""

    suffix = Path(file.filename or "profile.data").suffix or ".data"
    temp_path: str | None = None
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(content)
            temp_path = handle.name
        return analyze_profile(temp_path, params={"name": Path(file.filename or "MESA profile").stem})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.post("/batch_analyze")
async def batch_analyze_endpoint(file: Annotated[UploadFile, File(...)]) -> StreamingResponse:
    try:
        content = await file.read()
        frame = pd.read_csv(io.BytesIO(content))
        if len(frame) > 300000:
            raise HTTPException(status_code=413, detail="Batch limit is 300,000 stars")
        enriched = batch_analyze(frame)
        output = io.StringIO()
        enriched.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=stellar_batch_results.csv"},
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch analysis failed")
        raise HTTPException(status_code=500, detail="Batch analysis failed") from exc


def _build_report_pdf(result: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    factors = result["deviation_factors"]
    rows = [["Metric", "Value"]]
    rows.extend(
        [
            ["n_global", f"{result['global_fit']['n_global']:.4f}"],
            ["n_core", f"{result['piecewise_fit']['n_core']:.4f}"],
            ["n_rad", f"{result['piecewise_fit']['n_rad']:.4f}"],
            ["n_conv", f"{result['piecewise_fit']['n_conv']:.4f}"],
            ["delta_n_rad", f"{factors['delta_n_rad']:.4f}"],
            ["delta_n_mu", f"{factors['delta_n_mu']:.4f}"],
            ["delta_n_conv", f"{factors['delta_n_conv']:.4f}"],
            ["delta_n_nuc", f"{factors['delta_n_nuc']:.4f}"],
            ["delta_n_deg", f"{factors['delta_n_deg']:.4f}"],
            ["anomaly_score", f"{result['anomaly_score']:.4f}"],
            ["status", result["status"]],
        ]
    )
    table = Table(rows, colWidths=[180, 220])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
            ]
        )
    )
    story = [
        Paragraph("Stellar Polytropic Deviation Report", styles["Title"]),
        Paragraph(f"Star: {result['input']['name']}", styles["Normal"]),
        Spacer(1, 12),
        table,
        Spacer(1, 16),
        Paragraph(
            f"Health check: {result['status']} with residual {result['anomaly_score']:.4f}.",
            styles["Heading2"],
        ),
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


@app.get("/export/{star_id}")
def export_report(star_id: int) -> StreamingResponse:
    try:
        result = analyze_star({"name": f"Star {star_id}", "mass": 1.0, "teff": 5778.0, "metallicity": 0.0, "age": 4.6})
        pdf = _build_report_pdf(result)
        return StreamingResponse(
            io.BytesIO(pdf),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=star_{star_id}_report.pdf"},
        )
    except Exception as exc:
        logger.exception("PDF export failed")
        raise HTTPException(status_code=500, detail="PDF export failed") from exc
