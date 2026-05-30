"""HRV computation tests with synthetic R-peak trains."""

from __future__ import annotations

import numpy as np

from ecg_llm.signal.hrv import compute_hrv


def test_constant_rr_gives_zero_variability():
    fs = 360
    # Perfectly regular beats at 1 s spacing -> 60 bpm, zero SDNN/RMSSD.
    rpeaks = np.arange(0, 10 * fs, fs)
    hrv = compute_hrv(rpeaks, fs)
    assert round(hrv.mean_hr_bpm) == 60
    assert hrv.sdnn_ms == 0.0
    assert hrv.rmssd_ms == 0.0
    assert hrv.pnn50_pct == 0.0


def test_fewer_than_two_beats_is_safe():
    hrv = compute_hrv(np.array([5]), 360)
    assert hrv.n_beats == 1
    assert hrv.mean_hr_bpm == 0.0


def test_variable_rr_produces_positive_hrv():
    fs = 360
    # Alternating 0.8 s / 1.0 s intervals -> nonzero variability.
    intervals = []
    t = 0
    for i in range(20):
        intervals.append(t)
        t += int((0.8 if i % 2 == 0 else 1.0) * fs)
    hrv = compute_hrv(np.array(intervals), fs)
    assert hrv.sdnn_ms > 0
    assert hrv.rmssd_ms > 0
