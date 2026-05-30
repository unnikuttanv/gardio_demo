"""Synthetic ECG generator — the default, dependency-free sample source.

Produces a physiologically plausible single-lead ECG by summing Gaussian-shaped
P, Q, R, S, T deflections per beat. Heart rate varies beat-to-beat so the
downstream HRV metrics are non-trivial. Deterministic given a seed, so tests
can assert exact beat counts.
"""

from __future__ import annotations

import numpy as np

from ecg_llm.schemas import EcgSignal

# Relative timing (s) and amplitude (mV) of each wave within a beat, measured
# from the R peak. Rough but recognizable Lead-II morphology.
_WAVES = [
    # (center_offset_s, amplitude_mV, width_s)
    (-0.20, 0.10, 0.025),   # P
    (-0.025, -0.15, 0.012),  # Q
    (0.0, 1.0, 0.012),       # R
    (0.025, -0.25, 0.012),   # S
    (0.18, 0.30, 0.040),     # T
]


def _beat(t: np.ndarray, r_time: float) -> np.ndarray:
    wave = np.zeros_like(t)
    for offset, amp, width in _WAVES:
        center = r_time + offset
        wave += amp * np.exp(-0.5 * ((t - center) / width) ** 2)
    return wave


def generate_synthetic_ecg(
    duration_s: float = 30.0,
    fs: int = 360,
    mean_hr_bpm: float = 72.0,
    hr_std_bpm: float = 4.0,
    noise_mv: float = 0.02,
    baseline_wander_mv: float = 0.15,
    seed: int = 7,
    lead: str = "II",
) -> EcgSignal:
    """Generate a synthetic single-lead ECG.

    Args:
        duration_s: Recording length in seconds.
        fs: Sampling frequency (Hz).
        mean_hr_bpm: Target mean heart rate.
        hr_std_bpm: Beat-to-beat heart-rate variability (drives HRV metrics).
        noise_mv: White-noise amplitude.
        baseline_wander_mv: Low-frequency baseline drift amplitude.
        seed: RNG seed for reproducibility.
        lead: Lead label.

    Returns:
        An `EcgSignal` with `source="synthetic"`.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # Build R-peak times by drawing RR intervals around the target HR.
    mean_rr = 60.0 / mean_hr_bpm
    rr_std = (hr_std_bpm / mean_hr_bpm) * mean_rr
    r_times: list[float] = []
    clock = mean_rr  # first beat slightly in
    while clock < duration_s - mean_rr * 0.5:
        r_times.append(clock)
        rr = max(0.3, rng.normal(mean_rr, rr_std))
        clock += rr

    signal = np.zeros(n)
    for r_time in r_times:
        signal += _beat(t, r_time)

    # Baseline wander (~0.3 Hz) + white noise.
    signal += baseline_wander_mv * np.sin(2 * np.pi * 0.3 * t)
    signal += rng.normal(0.0, noise_mv, size=n)

    return EcgSignal(values=signal.tolist(), fs=fs, lead=lead, source="synthetic")
