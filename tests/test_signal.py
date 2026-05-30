"""Signal-processing tests — fully offline against the synthetic generator."""

from __future__ import annotations

import numpy as np

from ecg_llm.ingest import generate_synthetic_ecg
from ecg_llm.signal import clean, detect_rpeaks


def test_synthetic_is_reproducible():
    a = generate_synthetic_ecg(duration_s=10, seed=1)
    b = generate_synthetic_ecg(duration_s=10, seed=1)
    assert a.values == b.values
    assert a.fs == 360
    assert a.source == "synthetic"


def test_pan_tompkins_detects_expected_beat_count():
    # 30 s at 60 bpm -> ~30 beats. Allow generous tolerance for edge beats.
    sig = generate_synthetic_ecg(duration_s=30, mean_hr_bpm=60, hr_std_bpm=1.0, seed=3)
    filtered = clean(sig.values, sig.fs)
    rpeaks = detect_rpeaks(filtered, sig.fs)
    assert 26 <= len(rpeaks) <= 34


def test_rpeaks_are_spaced_by_refractory_period():
    sig = generate_synthetic_ecg(duration_s=20, mean_hr_bpm=75, seed=5)
    filtered = clean(sig.values, sig.fs)
    rpeaks = detect_rpeaks(filtered, sig.fs)
    diffs = np.diff(rpeaks)
    # No two detections closer than the 200 ms refractory period.
    assert (diffs >= int(0.2 * sig.fs)).all()


def test_clean_preserves_length():
    sig = generate_synthetic_ecg(duration_s=5, seed=2)
    filtered = clean(sig.values, sig.fs)
    assert len(filtered) == len(sig.values)
