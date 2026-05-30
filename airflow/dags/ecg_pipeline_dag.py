"""Airflow DAG for the ECG-LLM pipeline.

Four thin PythonOperator tasks call into `ecg_llm.pipeline` so the DAG runs the
exact same code as the FastAPI service. The package is mounted into the Airflow
containers (see docker-compose.yaml). XComs pass JSON between tasks.

    ingest -> extract_features -> generate_report -> persist
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "ecg-llm",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _ingest(**context):
    from ecg_llm.pipeline import ingest

    signal = ingest(source="synthetic", duration_s=30.0)
    context["ti"].xcom_push(key="signal", value=signal.model_dump_json())


def _extract_features(**context):
    from ecg_llm.features import build_features
    from ecg_llm.schemas import EcgSignal

    raw = context["ti"].xcom_pull(key="signal", task_ids="ingest")
    signal = EcgSignal.model_validate_json(raw)
    features = build_features(signal)
    context["ti"].xcom_push(key="features", value=features.model_dump_json())


def _generate_report(**context):
    from ecg_llm.report import generate_report
    from ecg_llm.schemas import FeatureSet

    raw = context["ti"].xcom_pull(key="features", task_ids="extract_features")
    features = FeatureSet.model_validate_json(raw)
    report = generate_report(features)
    context["ti"].xcom_push(key="report", value=report.model_dump_json())


def _persist(**context):
    raw = context["ti"].xcom_pull(key="report", task_ids="generate_report")
    out_dir = Path("/opt/airflow/data/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = context["ts_nodash"]
    out_path = out_dir / f"report_{stamp}.json"
    out_path.write_text(json.dumps(json.loads(raw), indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


with DAG(
    dag_id="ecg_pipeline",
    description="Synthetic ECG -> DSP features -> LLM narrative report.",
    default_args=DEFAULT_ARGS,
    schedule=None,  # trigger manually; set a cron string to schedule
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ecg", "llm", "signal-processing"],
) as dag:
    t_ingest = PythonOperator(task_id="ingest", python_callable=_ingest)
    t_features = PythonOperator(task_id="extract_features", python_callable=_extract_features)
    t_report = PythonOperator(task_id="generate_report", python_callable=_generate_report)
    t_persist = PythonOperator(task_id="persist", python_callable=_persist)

    t_ingest >> t_features >> t_report >> t_persist
