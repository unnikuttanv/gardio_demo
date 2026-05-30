# Image for the FastAPI API + Gradio demo (also used as the HF Spaces base).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for scipy/numpy wheels are bundled; keep image lean.
COPY pyproject.toml README.md ./
COPY ecg_llm ./ecg_llm
COPY api ./api
COPY app ./app

RUN pip install --upgrade pip && pip install .

EXPOSE 8000 7860

# Default: run the FastAPI service. Override CMD for the Gradio app.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
