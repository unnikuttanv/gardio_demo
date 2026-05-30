"""FastAPI service exposing the ECG-LLM pipeline.

Endpoints:
    GET  /health   -> liveness + whether the LLM is mocked.
    POST /analyze  -> run the pipeline and return an EcgReport.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ecg_llm.config import get_settings
from ecg_llm.pipeline import run_pipeline
from ecg_llm.schemas import EcgReport

app = FastAPI(
    title="ECG-LLM-Analyst",
    description="Classical ECG signal processing piped into an LLM for clinical narratives.",
    version="0.1.0",
)


class AnalyzeRequest(BaseModel):
    source: str = Field("synthetic", description="'synthetic' or a PhysioNet record name.")
    duration_s: float = Field(30.0, description="Synthetic recording length (ignored for real records).")
    mean_hr_bpm: float = Field(72.0, description="Synthetic target mean heart rate.")
    pn_dir: str | None = Field(None, description="PhysioNet database dir for real records, e.g. 'mitdb'.")


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.llm_model,
        "llm_mocked": settings.use_mock_llm,
    }


@app.post("/analyze", response_model=EcgReport)
def analyze(req: AnalyzeRequest) -> EcgReport:
    if req.source == "synthetic":
        return run_pipeline(
            source="synthetic",
            duration_s=req.duration_s,
            mean_hr_bpm=req.mean_hr_bpm,
        )
    kwargs = {"pn_dir": req.pn_dir} if req.pn_dir else {}
    return run_pipeline(source=req.source, **kwargs)
