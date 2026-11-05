"""Frequency display helpers for logs, summaries, and plots."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

FrequencyLike = Union[float, np.ndarray, list, tuple]


def fmt_hz(value: float, *, na: str = "—") -> str:
    """Format a scalar frequency in Hz with automatic GHz/MHz/kHz/Hz scaling."""
    if value != value:  # NaN
        return na
    if abs(value) >= 1e9:
        return f"{value / 1e9:.6f} GHz"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f} MHz"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f} kHz"
    return f"{value:.3g} Hz"


def scale_frequency_hz(values: FrequencyLike) -> Tuple[float, str]:
    """Return ``(divisor, unit_label)`` to express Hz *values* for plotting or axes."""
    arr = np.asarray(values, dtype=float)
    max_abs = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
    if max_abs >= 1e9:
        return 1e9, "GHz"
    if max_abs >= 1e6:
        return 1e6, "MHz"
    if max_abs >= 1e3:
        return 1e3, "kHz"
    return 1.0, "Hz"
