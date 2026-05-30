"""PhysioNet WFDB record loading — the real-data path.

Uses the `wfdb` package to read or download records. Kept separate from the
synthetic generator so the rest of the pipeline is agnostic to data source:
both return an `EcgSignal`.
"""

from __future__ import annotations

import numpy as np

from ecg_llm.schemas import EcgSignal


def load_record(
    record_name: str,
    pn_dir: str | None = None,
    channel: int = 0,
) -> EcgSignal:
    """Load a WFDB record into an `EcgSignal`.

    Args:
        record_name: Record name (without extension), e.g. "100".
        pn_dir: PhysioNet database dir to download from, e.g. "mitdb". When
            None, `record_name` is treated as a local path.
        channel: Which signal channel/lead to extract.

    Returns:
        An `EcgSignal` with `source="physionet:<record>"`.

    Raises:
        ImportError: If `wfdb` is not installed.
    """
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "wfdb is required for PhysioNet loading. Install with `pip install wfdb`."
        ) from exc

    if pn_dir:
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    else:
        record = wfdb.rdrecord(record_name)

    signal = np.asarray(record.p_signal[:, channel], dtype=float)
    fs = int(record.fs)
    lead = record.sig_name[channel] if record.sig_name else f"ch{channel}"

    return EcgSignal(
        values=signal.tolist(),
        fs=fs,
        lead=lead,
        source=f"physionet:{record_name}",
    )
