"""Weekly trend summaries — LLM reasoning over a user's multi-day features.

Mirrors the single-report flow: call the configured model, but degrade
gracefully to a deterministic mock on any failure or when no key is configured.
"""

from __future__ import annotations

import logging

from ecg_llm.config import Settings, get_settings
from ecg_llm.report import llm
from ecg_llm.report.prompts import WEEKLY_SYSTEM_PROMPT, build_weekly_prompt
from ecg_llm.schemas import DailyRecord

logger = logging.getLogger(__name__)


def _trend(values: list[float]) -> str:
    """Crude direction label from first vs last value."""
    if len(values) < 2:
        return "stable"
    delta = values[-1] - values[0]
    spread = max(abs(values[0]), 1.0)
    if delta > 0.08 * spread:
        return "rising"
    if delta < -0.08 * spread:
        return "falling"
    return "stable"


def mock_weekly_summary(name: str, profile: str, daily: list[DailyRecord]) -> str:
    """Deterministic, trend-grounded summary used when no model is available."""
    hrs = [r.features.mean_hr_bpm for r in daily]
    sdnns = [r.features.sdnn_ms for r in daily]
    hr_dir = _trend(hrs)
    hrv_dir = _trend(sdnns)
    all_flags = sorted({f for r in daily for f in r.features.flags})

    return (
        f"Heart-rate trend: Over {len(daily)} days, {name}'s mean heart rate moved "
        f"from {hrs[0]:.0f} to {hrs[-1]:.0f} bpm ({hr_dir}). "
        f"Daily means ranged {min(hrs):.0f}-{max(hrs):.0f} bpm.\n\n"
        f"HRV trend: SDNN went from {sdnns[0]:.0f} to {sdnns[-1]:.0f} ms ({hrv_dir}), "
        f"suggesting a {hrv_dir} pattern in autonomic variability over the window.\n\n"
        f"Standout signals: flags observed across the week were "
        f"{', '.join(all_flags) if all_flags else 'none'}. Profile context: {profile}.\n\n"
        f"Disclaimer: Automated, non-diagnostic trend summary — requires clinician confirmation."
    )


def generate_weekly_summary(
    name: str,
    profile: str,
    daily: list[DailyRecord],
    settings: Settings | None = None,
) -> tuple[str, str, bool]:
    """Generate a weekly trend summary.

    Returns:
        (summary_text, model_name, is_mock).
    """
    settings = settings or get_settings()
    if settings.use_mock_llm:
        return mock_weekly_summary(name, profile, daily), "mock", True

    messages = [
        {"role": "system", "content": WEEKLY_SYSTEM_PROMPT},
        {"role": "user", "content": build_weekly_prompt(name, profile, daily)},
    ]
    try:
        text = llm.complete(messages, settings)
        return text, settings.llm_model, False
    except Exception as exc:  # noqa: BLE001 - intentional broad fallback
        logger.warning("Weekly summary LLM call failed (%s); using mock.", exc)
        return mock_weekly_summary(name, profile, daily), "mock", True
