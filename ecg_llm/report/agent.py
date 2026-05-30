"""LangGraph agent: FeatureSet -> EcgReport.

A small three-node graph keeps the generative flow explicit and inspectable:

    compose -> narrate -> structure

`compose`  builds the prompt messages from features.
`narrate`  calls the model (or the mock) to produce the narrative.
`structure` distills structured findings from the features.

Using a graph (rather than a single function) makes it trivial to add nodes
later — e.g. a critique/verify step or retrieval — without reshaping callers.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from ecg_llm.config import Settings, get_settings
from ecg_llm.report import llm
from ecg_llm.report.prompts import SYSTEM_PROMPT, build_user_prompt
from ecg_llm.schemas import EcgReport, FeatureSet

logger = logging.getLogger(__name__)


class _State(TypedDict, total=False):
    features: FeatureSet
    settings: Settings
    messages: list[dict]
    narrative: str
    findings: list[str]
    is_mock: bool


def _compose(state: _State) -> _State:
    features = state["features"]
    state["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(features)},
    ]
    return state


def _narrate(state: _State) -> _State:
    settings = state["settings"]
    if settings.use_mock_llm:
        state["narrative"] = llm.mock_narrative(state["features"])
        state["is_mock"] = True
        return state

    # Real model configured — call it, but degrade gracefully to the mock on any
    # failure (bad key, rate limit, network) so the pipeline never hard-fails.
    try:
        state["narrative"] = llm.complete(state["messages"], settings)
        state["is_mock"] = False
    except Exception as exc:  # noqa: BLE001 - intentional broad fallback
        logger.warning("LLM call failed (%s); falling back to mock narrative.", exc)
        state["narrative"] = llm.mock_narrative(state["features"])
        state["is_mock"] = True
    return state


def _structure(state: _State) -> _State:
    f = state["features"]
    findings = [
        f"{f.n_beats} beats over {f.duration_s:.0f}s",
        f"mean HR {f.mean_hr_bpm:.0f} bpm",
        f"SDNN {f.sdnn_ms:.0f} ms / RMSSD {f.rmssd_ms:.0f} ms / pNN50 {f.pnn50_pct:.0f}%",
    ]
    findings.extend(f.flags)
    state["findings"] = findings
    return state


def _build_graph():
    graph = StateGraph(_State)
    graph.add_node("compose", _compose)
    graph.add_node("narrate", _narrate)
    graph.add_node("structure", _structure)
    graph.set_entry_point("compose")
    graph.add_edge("compose", "narrate")
    graph.add_edge("narrate", "structure")
    graph.add_edge("structure", END)
    return graph.compile()


# Compiled once at import — the graph is stateless across runs.
_GRAPH = _build_graph()


def generate_report(features: FeatureSet, settings: Settings | None = None) -> EcgReport:
    """Run the report agent and return an EcgReport.

    Args:
        features: Interpreted features from the deterministic pipeline.
        settings: Optional override; defaults to env-derived settings.

    Returns:
        A populated `EcgReport` (narrative + structured findings + provenance).
    """
    settings = settings or get_settings()
    result = _GRAPH.invoke({"features": features, "settings": settings})
    return EcgReport(
        narrative=result["narrative"],
        findings=result["findings"],
        features=features,
        model="mock" if result["is_mock"] else settings.llm_model,
        is_mock=result["is_mock"],
    )
