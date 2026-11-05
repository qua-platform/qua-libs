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
) -> np.ndarray:
    """Build a validated linear ramp-duration sweep in ns."""
    ramp_min = int(ramp_duration_min)
    ramp_max = int(ramp_duration_max)
    ramp_step = int(ramp_duration_step)
    ramp_duration_array = np.arange(ramp_min, ramp_max, ramp_step, dtype=int)
    if ramp_duration_array.size < 1:
        raise ValueError("Empty ramp duration sweep: require ramp_duration_min < ramp_duration_max with positive step.")
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
    )
