# ECG-LLM-Analyst — Scaffold Design

**Date:** 2026-05-30
**Status:** Approved (design), pending implementation

## Goal

Agentic ECG report generator: load an ECG signal, run classical signal
processing (Pan-Tompkins QRS detection, HRV), then pipe the extracted features
into an LLM to produce a clinical narrative report. Demoable via FastAPI + Gradio,
orchestrated via Airflow.

## Scope decisions (locked)

- **Scaffold depth:** Runnable skeleton — full structure + a *working* DSP core
  and an end-to-end pipeline that runs on clone with no API key (mock LLM fallback).
- **Airflow:** Full Docker-compose stack with a working DAG.
- **LLM provider:** Grok (xAI) by default via LiteLLM (`xai/grok-4`), configurable
  to any LiteLLM model. Auto-falls-back to a deterministic mock provider when no
  `XAI_API_KEY` is present.
- **Sample data:** Synthetic ECG generated in-code — no external data dependency,
  no licensing questions. (Real PhysioNet loading is still supported via `ingest/`.)
- **Git:** No commits during this scaffold pass.

## Directory layout

```
ecgllm/
├── ecg_llm/                      # importable, testable package
│   ├── ingest/
│   │   ├── physionet.py          # WFDB record loading (real data path)
│   │   └── synthetic.py          # synthetic ECG generator (default sample)
│   ├── signal/
│   │   ├── preprocess.py         # bandpass + baseline-wander removal
│   │   ├── pan_tompkins.py       # QRS detection → R-peak indices
│   │   └── hrv.py                # RR intervals → HRV features
│   ├── features/
│   │   └── extract.py            # signal/* → FeatureSet
│   ├── report/
│   │   ├── llm.py                # LiteLLM client (xai/grok default + mock)
│   │   ├── agent.py              # LangGraph flow: features → narrative
│   │   └── prompts.py            # prompt templates
│   ├── pipeline.py               # ingest→features→report orchestration
│   ├── schemas.py                # Pydantic: FeatureSet, EcgReport
│   └── config.py                 # pydantic-settings env config
├── api/main.py                   # FastAPI: POST /analyze, GET /health
├── app/gradio_app.py             # Gradio demo (HF Spaces entrypoint)
├── airflow/
│   ├── docker-compose.yaml
│   └── dags/ecg_pipeline_dag.py
├── tests/                        # pytest, offline (synthetic + mock)
├── data/sample/                  # generated synthetic record artifacts (gitignored)
├── pyproject.toml
├── .env.example
├── Dockerfile
├── README.md
└── .gitignore
```

## Boundaries

- `signal/` is pure NumPy/SciPy — no LLM, no I/O. Unit-tested against known peaks.
- `report/` is the only module touching the LLM.
- `FeatureSet` (Pydantic) is the contract between deterministic and generative halves.
- `pipeline.py` is callable standalone AND from the Airflow DAG (orchestration-only).

## Data flow

```
record → ingest (signal, fs, leads)
       → preprocess.clean (bandpass 0.5–40 Hz, baseline removed)
       → pan_tompkins.detect (R-peak indices)
       → hrv.compute (RR intervals, SDNN, RMSSD, pNN50, mean HR)
       → features.extract.build (FeatureSet + rule-based flags)
       → report.agent.run (EcgReport: narrative + structured findings)
```

`FeatureSet`: mean HR, SDNN, RMSSD, pNN50, RR count, duration, fs, lead, and
rule-based flags (tachycardia/bradycardia/etc.) so the LLM gets interpreted numbers.
`EcgReport`: narrative + structured findings + `is_mock` flag (honest about whether
a real model ran).

## Config (pydantic-settings, `.env`)

- `LLM_MODEL` (default `xai/grok-4`)
- `XAI_API_KEY` (absent → mock provider)
- `LLM_TEMPERATURE`, `DATA_DIR`, `SAMPLE_RECORD`

## Airflow

Standard docker-compose Airflow. `ecg_pipeline_dag.py`: `ingest → extract_features
→ generate_report → persist`, each a thin PythonOperator calling `ecg_llm.pipeline`.
Package mounted into containers so the DAG imports the same code as the API.

## Testing

- `signal/` tests: offline against synthetic record, assert R-peak count / HR range.
- `report/` tests: against mock provider (no network/key).
- One end-to-end smoke test through `pipeline.py` with the mock.

## Dependencies

wfdb, numpy, scipy, litellm, langgraph, fastapi, uvicorn, gradio,
pydantic, pydantic-settings, pytest, ruff.
