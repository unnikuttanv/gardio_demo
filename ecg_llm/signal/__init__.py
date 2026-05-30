"""Classical ECG signal processing — pure NumPy/SciPy, no LLM, no I/O.

This package is the deterministic core. Everything here is unit-testable against
known inputs.
"""

from ecg_llm.signal.hrv import compute_hrv
from ecg_llm.signal.pan_tompkins import detect_rpeaks
from ecg_llm.signal.preprocess import clean

__all__ = ["clean", "detect_rpeaks", "compute_hrv"]
