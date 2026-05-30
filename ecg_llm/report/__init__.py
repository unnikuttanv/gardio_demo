"""Generative report layer — the only package that touches the LLM."""

from ecg_llm.report.agent import generate_report

__all__ = ["generate_report"]
