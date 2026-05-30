"""Pan-Tompkins QRS detection.

Implements the classic pipeline: derivative -> squaring -> moving-window
integration -> adaptive thresholding, with R-peak refinement against the
filtered signal. Returns sample indices of detected R peaks.

Reference: Pan J, Tompkins WJ. "A Real-Time QRS Detection Algorithm."
IEEE Trans. Biomed. Eng. 1985.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def detect_rpeaks(filtered: np.ndarray, fs: int) -> np.ndarray:
    """Detect R-peak indices in a band-pass-filtered ECG.

    Args:
        filtered: Band-pass-filtered ECG (see preprocess.clean).
        fs: Sampling frequency (Hz).

    Returns:
        Sorted ndarray of R-peak sample indices.
    """
    signal = np.asarray(filtered, dtype=float)
    if signal.size < fs // 2:
        return np.array([], dtype=int)

    # 1. Derivative — emphasize the steep QRS slope.
    derivative = np.ediff1d(signal, to_begin=0.0)

    # 2. Squaring — make everything positive and amplify large slopes.
    squared = derivative**2

    # 3. Moving-window integration (~150 ms window).
    win = max(1, int(0.150 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    # 4. Candidate peaks on the integrated signal.
    #    Refractory period ~200 ms (physiological QRS spacing limit).
    min_distance = int(0.200 * fs)
    threshold = 0.3 * np.mean(integrated) + 0.05 * np.max(integrated)
    candidates, _ = find_peaks(integrated, distance=min_distance, height=threshold)

    # 5. Refine: snap each candidate to the local max of the filtered signal
    #    within a small search window (the integration window shifts peaks).
    search = max(1, int(0.050 * fs))
    rpeaks = []
    for c in candidates:
        lo = max(0, c - search)
        hi = min(len(signal), c + search + 1)
        rpeaks.append(lo + int(np.argmax(signal[lo:hi])))

    return np.unique(np.asarray(rpeaks, dtype=int))
