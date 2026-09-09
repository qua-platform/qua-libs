from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qualibrate.core import QualibrationNode


__all__ = [
    "build_ramp_duration_sweep",
    "validate_and_build_ramp_sweep",
]


def build_ramp_duration_sweep(
    ramp_duration_min: int,
    ramp_duration_max: int,
    ramp_duration_step: int,
    *,
    log_scale: bool = False,
) -> np.ndarray:
    """Build a validated ramp-duration sweep in ns.

    For linear sweeps this returns ``np.arange(min, max, step)``.
    For logarithmic sweeps this returns integer-valued ``np.logspace(...)`` samples
    spanning the same range, using the linear-sweep point count as the requested
    resolution and de-duplicating any repeated integer samples after rounding.
    """
    ramp_min = int(ramp_duration_min)
    ramp_max = int(ramp_duration_max)
    ramp_step = int(ramp_duration_step)

    linear_count = len(np.arange(ramp_min, ramp_max, ramp_step, dtype=int))
    if linear_count < 1:
        raise ValueError("Empty ramp duration sweep: require ramp_duration_min < ramp_duration_max with positive step.")

    if not log_scale:
        return np.arange(ramp_min, ramp_max, ramp_step, dtype=int)

    if ramp_min <= 0 or ramp_max <= 0:
        raise ValueError("Logarithmic ramp-duration sweeps require positive ramp bounds.")

    if linear_count == 1:
        return np.array([ramp_min], dtype=int)

    ramp_duration_array = np.logspace(
        np.log10(ramp_min),
        np.log10(ramp_max),
        linear_count,
        dtype=int,
        endpoint=True,
    )
    ramp_duration_array = np.unique(ramp_duration_array)
    if ramp_duration_array.size < 1:
        raise ValueError("Empty ramp duration sweep after building logarithmic samples.")

    return ramp_duration_array


def validate_and_build_ramp_sweep(node: "QualibrationNode") -> np.ndarray:
    """Validate ramp settings from ``node.parameters`` and return the ramp sweep."""
    ramp_min = int(node.parameters.ramp_duration_min)
    ramp_max = int(node.parameters.ramp_duration_max)
    ramp_step = int(node.parameters.ramp_duration_step)

    if ramp_min % 4 != 0 or ramp_max % 4 != 0 or ramp_step % 4 != 0:
        raise ValueError(
            f"Ramp settings must be divisible by 4. Got min={ramp_min}, max={ramp_max}, step={ramp_step}"
        )

    return build_ramp_duration_sweep(
        ramp_min,
        ramp_max,
        ramp_step,
        log_scale=bool(getattr(node.parameters, "ramp_log_scale", False)),
    )
