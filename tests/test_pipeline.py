"""End-to-end smoke tests through the pipeline using the mock LLM."""

from __future__ import annotations

from ecg_llm.config import Settings
from ecg_llm.features import build_features
from ecg_llm.ingest import generate_synthetic_ecg
from ecg_llm.pipeline import run_pipeline
from ecg_llm.report import generate_report


def _mock_settings() -> Settings:
    # Force the mock path regardless of any .env / host environment: ignore the
    # .env file and clear every provider key.
    return Settings(_env_file=None, groq_api_key="", xai_api_key="", openai_api_key="")


def test_features_have_expected_flags():
    sig = generate_synthetic_ecg(duration_s=20, mean_hr_bpm=72, seed=4)
    features = build_features(sig)
    assert features.n_beats > 10
    assert "normal_rate" in features.flags


def test_report_uses_mock_without_key():
    sig = generate_synthetic_ecg(duration_s=15, seed=6)
    features = build_features(sig)
    report = generate_report(features, settings=_mock_settings())
    assert report.is_mock is True
    assert report.model == "mock"
    assert "Disclaimer" in report.narrative
    assert report.findings


def test_run_pipeline_end_to_end():
    report = run_pipeline(
        source="synthetic",
        settings=_mock_settings(),
        duration_s=20,
        mean_hr_bpm=110,
    )
    assert report.features.n_beats > 10
    assert "tachycardia" in report.features.flags
    assert report.narrative
