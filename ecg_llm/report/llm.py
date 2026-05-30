"""LLM access via LiteLLM, with a deterministic mock fallback.

The real path routes through LiteLLM so any provider works by changing
`LLM_MODEL` (default `xai/grok-4`). When no credential is configured,
`mock_narrative` produces a deterministic, feature-grounded report so the
demo runs offline with no API key.
"""

from __future__ import annotations

from ecg_llm.config import Settings
from ecg_llm.schemas import FeatureSet


def complete(messages: list[dict], settings: Settings) -> str:
    """Call the configured model through LiteLLM and return the text response.

    Args:
        messages: OpenAI-style chat messages.
        settings: Runtime settings (model, key, temperature).

    Returns:
        The assistant's text content.

    Raises:
        ImportError: If `litellm` is not installed.
    """
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "litellm is required for real LLM calls. Install with `pip install litellm`."
        ) from exc

    response = litellm.completion(
        model=settings.llm_model,
        messages=messages,
        api_key=settings.api_key or None,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return response["choices"][0]["message"]["content"].strip()


def mock_narrative(features: FeatureSet) -> str:
    """Deterministic, feature-grounded narrative used when no model is configured."""
    flag_text = {
        "tachycardia": "an elevated heart rate consistent with tachycardia",
        "bradycardia": "a slow heart rate consistent with bradycardia",
        "normal_rate": "a heart rate within the normal resting range",
        "low_hrv": "reduced heart-rate variability",
        "high_hrv": "elevated heart-rate variability",
        "insufficient_beats": "too few detected beats for reliable analysis",
    }
    described = [flag_text.get(f, f) for f in features.flags]
    rhythm_line = "; ".join(described) if described else "no notable rhythm flags"

    return (
        f"Rhythm & rate: Over {features.duration_s:.0f} s of lead {features.lead}, "
        f"{features.n_beats} beats were detected at a mean rate of "
        f"{features.mean_hr_bpm:.0f} bpm (range {features.min_hr_bpm:.0f}-"
        f"{features.max_hr_bpm:.0f} bpm). The recording shows {rhythm_line}.\n\n"
        f"HRV interpretation: Time-domain variability measures are SDNN "
        f"{features.sdnn_ms:.0f} ms, RMSSD {features.rmssd_ms:.0f} ms, and pNN50 "
        f"{features.pnn50_pct:.0f}%. These reflect beat-to-beat autonomic modulation "
        f"and should be read in clinical context.\n\n"
        f"Findings: Automated single-lead analysis; morphology beyond rate/variability "
        f"was not assessed. Source: {features.source}.\n\n"
        f"Disclaimer: Automated, non-diagnostic summary — requires clinician confirmation."
    )
