"""ECG preprocessing: bandpass filtering and baseline-wander removal."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass(signal: np.ndarray, fs: int, low: float, high: float, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = min(high / nyq, 0.99)
    b, a = butter(order, [low_n, high_n], btype="band")
    # filtfilt needs padding shorter than the signal.
    padlen = min(3 * max(len(a), len(b)), len(signal) - 1)
    return filtfilt(b, a, signal, padlen=max(padlen, 0))


def clean(values: list[float] | np.ndarray, fs: int, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Bandpass-filter an ECG to remove baseline wander and high-frequency noise.

    Args:
        values: Raw amplitude samples.
        fs: Sampling frequency (Hz).
        low: Low cutoff (Hz) — removes baseline wander.
        high: High cutoff (Hz) — removes muscle/EMG noise.

    Returns:
        Filtered signal as a float ndarray (same length as input).
    """
    signal = np.asarray(values, dtype=float)
    if signal.size < 10:
        return signal
    return _bandpass(signal, fs, low, high)
