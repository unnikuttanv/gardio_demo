"""End-to-end orchestration: ingest -> features -> report.

Callable standalone (`python -m ecg_llm.pipeline`), from the FastAPI service,
and from the Airflow DAG. The DAG calls the individual step functions; everyone
else calls `run_pipeline`.
"""

from __future__ import annotations

import argparse
import json

from ecg_llm.config import Settings, get_settings
from ecg_llm.features import build_features
from ecg_llm.ingest import generate_synthetic_ecg
from ecg_llm.ingest.physionet import load_record
from ecg_llm.report import generate_report
from ecg_llm.schemas import EcgReport, EcgSignal, FeatureSet


def ingest(source: str = "synthetic", **kwargs) -> EcgSignal:
    """Load an ECG. `source="synthetic"` generates one; otherwise treat as a record."""
    if source == "synthetic":
        return generate_synthetic_ecg(**kwargs)
    return load_record(source, **kwargs)


def extract_features(signal: EcgSignal) -> FeatureSet:
    """Run the deterministic signal-processing pipeline."""
    return build_features(signal)


def report(features: FeatureSet, settings: Settings | None = None) -> EcgReport:
    """Run the generative report agent."""
    return generate_report(features, settings)


def run_pipeline(source: str = "synthetic", settings: Settings | None = None, **kwargs) -> EcgReport:
    """Full pipeline: ingest -> features -> report.

    Args:
        source: "synthetic" or a record name/path.
        settings: Optional settings override.
        **kwargs: Forwarded to `ingest` (e.g. duration_s, fs, pn_dir).

    Returns:
        The final `EcgReport`.
    """
    settings = settings or get_settings()
    signal = ingest(source, **kwargs)
    features = extract_features(signal)
    return report(features, settings)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the ECG-LLM pipeline.")
    parser.add_argument("--source", default="synthetic", help="'synthetic' or a record name.")
    parser.add_argument("--duration", type=float, default=30.0, help="Synthetic duration (s).")
    args = parser.parse_args()

    kwargs = {"duration_s": args.duration} if args.source == "synthetic" else {}
    result = run_pipeline(source=args.source, **kwargs)
    print(json.dumps(result.model_dump(), indent=2, default=float))


if __name__ == "__main__":
    _main()
