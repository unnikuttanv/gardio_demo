"""Heart-rate variability (time-domain) from R-peak locations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HrvResult:
    """Time-domain HRV metrics plus the underlying instantaneous heart rates."""

    n_beats: int
    mean_hr_bpm: float
    min_hr_bpm: float
    max_hr_bpm: float
    sdnn_ms: float
    rmssd_ms: float
    pnn50_pct: float


def compute_hrv(rpeaks: np.ndarray, fs: int) -> HrvResult:
    """Compute time-domain HRV metrics from R-peak indices.

    Args:
        rpeaks: Sample indices of R peaks.
        fs: Sampling frequency (Hz).

    Returns:
        An `HrvResult`. With fewer than 2 beats all metrics are 0.
    """
    rpeaks = np.asarray(rpeaks, dtype=int)
    n_beats = int(rpeaks.size)
    if n_beats < 2:
        return HrvResult(n_beats, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # RR intervals in milliseconds.
    rr_ms = np.diff(rpeaks) / fs * 1000.0
    inst_hr = 60000.0 / rr_ms

    sdnn = float(np.std(rr_ms, ddof=1)) if rr_ms.size > 1 else 0.0
    successive_diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(successive_diff**2))) if successive_diff.size else 0.0
    pnn50 = (
        float(np.mean(np.abs(successive_diff) > 50.0) * 100.0)
        if successive_diff.size
        else 0.0
    )

    return HrvResult(
        n_beats=n_beats,
        mean_hr_bpm=float(np.mean(inst_hr)),
        min_hr_bpm=float(np.min(inst_hr)),
        max_hr_bpm=float(np.max(inst_hr)),
        sdnn_ms=sdnn,
        rmssd_ms=rmssd,
        pnn50_pct=pnn50,
    )
