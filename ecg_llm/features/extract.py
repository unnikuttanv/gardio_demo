"""Turn a raw EcgSignal into an interpreted FeatureSet.

Orchestrates preprocess -> Pan-Tompkins -> HRV, then attaches simple rule-based
flags so the LLM receives interpreted numbers rather than raw statistics.
"""

from __future__ import annotations

from ecg_llm.schemas import EcgSignal, FeatureSet
from ecg_llm.signal import clean, compute_hrv, detect_rpeaks

# Coarse clinical thresholds — illustrative, not diagnostic.
_TACHY_BPM = 100.0
_BRADY_BPM = 60.0
_HIGH_SDNN_MS = 100.0
_LOW_SDNN_MS = 20.0


def _derive_flags(mean_hr: float, sdnn_ms: float, n_beats: int) -> list[str]:
    flags: list[str] = []
    if n_beats < 2:
        flags.append("insufficient_beats")
        return flags
    if mean_hr > _TACHY_BPM:
        flags.append("tachycardia")
    elif mean_hr < _BRADY_BPM:
        flags.append("bradycardia")
    else:
        flags.append("normal_rate")
    if sdnn_ms < _LOW_SDNN_MS:
        flags.append("low_hrv")
    elif sdnn_ms > _HIGH_SDNN_MS:
        flags.append("high_hrv")
    return flags


def build_features(signal: EcgSignal) -> FeatureSet:
    """Run the deterministic pipeline and return a FeatureSet.

    Args:
        signal: A loaded ECG (synthetic or PhysioNet).

    Returns:
        A populated `FeatureSet` with rhythm, HRV, and rule-based flags.
    """
    filtered = clean(signal.values, signal.fs)
    rpeaks = detect_rpeaks(filtered, signal.fs)
    hrv = compute_hrv(rpeaks, signal.fs)

    flags = _derive_flags(hrv.mean_hr_bpm, hrv.sdnn_ms, hrv.n_beats)

    return FeatureSet(
        source=signal.source,
        lead=signal.lead,
        fs=signal.fs,
        duration_s=round(signal.duration_s, 2),
        n_beats=hrv.n_beats,
        mean_hr_bpm=round(hrv.mean_hr_bpm, 1),
        min_hr_bpm=round(hrv.min_hr_bpm, 1),
        max_hr_bpm=round(hrv.max_hr_bpm, 1),
        sdnn_ms=round(hrv.sdnn_ms, 1),
        rmssd_ms=round(hrv.rmssd_ms, 1),
        pnn50_pct=round(hrv.pnn50_pct, 1),
        flags=flags,
    )
