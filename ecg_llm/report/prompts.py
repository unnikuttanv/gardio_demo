"""Prompt templates for the ECG narrative report."""

from __future__ import annotations

from ecg_llm.schemas import DailyRecord, FeatureSet

SYSTEM_PROMPT = (
    "You are a cardiology assistant that writes concise, structured ECG narrative "
    "reports for clinician review. You are given pre-computed signal-processing "
    "features (heart rate, HRV, rhythm flags) from a single-lead ECG. Base your "
    "report ONLY on the provided features. Do not invent measurements. Always end "
    "with the disclaimer that this is an automated, non-diagnostic summary requiring "
    "clinician confirmation."
)


def build_user_prompt(features: FeatureSet) -> str:
    """Render a FeatureSet into the user prompt."""
    flags = ", ".join(features.flags) if features.flags else "none"
    return (
        "Generate an ECG narrative report from these features:\n\n"
        f"- Source: {features.source}\n"
        f"- Lead: {features.lead}\n"
        f"- Sampling rate: {features.fs} Hz\n"
        f"- Duration: {features.duration_s} s\n"
        f"- Beats detected: {features.n_beats}\n"
        f"- Mean HR: {features.mean_hr_bpm} bpm "
        f"(min {features.min_hr_bpm}, max {features.max_hr_bpm})\n"
        f"- SDNN: {features.sdnn_ms} ms\n"
        f"- RMSSD: {features.rmssd_ms} ms\n"
        f"- pNN50: {features.pnn50_pct} %\n"
        f"- Rule-based flags: {flags}\n\n"
        "Write 2-4 short paragraphs: (1) rhythm and rate summary, "
        "(2) HRV interpretation, (3) noteworthy findings or caveats. "
        "Then a one-line non-diagnostic disclaimer."
    )


WEEKLY_SYSTEM_PROMPT = (
    "You are a cardiology assistant writing a concise weekly trend summary for a "
    "single remotely-monitored patient. You are given pre-computed per-day ECG "
    "features (heart rate and HRV) across several days. Focus on the TREND over "
    "time: is heart rate rising/falling/stable, is HRV improving or deteriorating, "
    "and are there days that stand out. Base everything ONLY on the numbers given. "
    "Do not invent measurements. End with a one-line non-diagnostic disclaimer."
)


def build_weekly_prompt(name: str, profile: str, daily: list[DailyRecord]) -> str:
    """Render a user's multi-day records into a weekly-summary prompt."""
    lines = [
        f"Patient: {name} ({profile})",
        f"Monitoring window: {len(daily)} days",
        "",
        "Per-day features (date | mean HR bpm | SDNN ms | RMSSD ms | flags):",
    ]
    for rec in daily:
        f = rec.features
        flags = ",".join(f.flags) if f.flags else "none"
        lines.append(
            f"- {rec.date} | {f.mean_hr_bpm} | {f.sdnn_ms} | {f.rmssd_ms} | {flags}"
        )
    lines += [
        "",
        "Write a 2-3 paragraph weekly summary: (1) heart-rate trend across the days, "
        "(2) HRV trend and what it suggests about autonomic state, (3) any standout "
        "days or a recommendation to review. Then a one-line non-diagnostic disclaimer.",
    ]
    return "\n".join(lines)
