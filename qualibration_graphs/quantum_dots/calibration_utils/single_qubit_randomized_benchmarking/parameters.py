"""Parameters for single-qubit randomized benchmarking (node 14).

The PPU-optimized RB experiment measures the average Clifford gate fidelity
by playing random sequences of Clifford gates at increasing circuit depths
and fitting the survival probability to an exponential decay.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for 14_single_qubit_randomized_benchmarking."""

    num_circuits_per_length: int = 50
    """Number of random circuits per depth. Default is 50."""
    num_shots: int = 400
    """Number of repetitions (shots) per circuit. Default is 400."""
    max_circuit_depth: int = 256
    """Maximum circuit depth (total Clifford count). Default is 256."""
    delta_clifford: int = 20
    """Step between depths in linear scale mode. Default is 20."""
    log_scale: bool = True
    """If True, use log-scale depths: 2, 4, 8, 16, ... up to max_circuit_depth. Default is True."""
    seed: Optional[int] = None
    """Seed for the QUA pseudo-random number generator. Default is None (random)."""
    gap_wait_time_in_ns: int = 400_000
    """Initialization hold time in nanoseconds (e.g. 400 µs). Default is 400_000."""
    operation_x90: str = "x90"
    """Name of the π/2 X rotation operation on the xy channel. Default is 'x90'."""
    operation_x180: str = "x180"
    """Name of the π X rotation operation on the xy channel. Default is 'x180'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Full parameter set for single-qubit randomized benchmarking."""

    def get_depths(self) -> np.ndarray:
        """Generate an array of circuit depths based on the parameter configuration.

        Depth *d* means *d-1* random Cliffords + 1 recovery (inverse) gate
        = *d* total Cliffords.  This is the standard RB convention.

        - If ``log_scale`` is True, depths follow a power-of-two progression:
          2, 4, 8, 16, 32, ... up to ``max_circuit_depth``.
        - If ``log_scale`` is False, depths are linearly spaced using
          ``delta_clifford`` until ``max_circuit_depth``.  The first value
          is always set to 1.

        Returns
        -------
        numpy.ndarray
            Sorted array of circuit depths (integers).
        """
        if self.log_scale:
            depths = []
            current_depth = 2
            while current_depth <= self.max_circuit_depth:
                depths.append(current_depth)
                current_depth *= 2
            return np.array(depths, dtype=int)

        assert (
            self.max_circuit_depth / self.delta_clifford
        ).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
        depths_arr = np.arange(0, self.max_circuit_depth + 0.1, self.delta_clifford, dtype=int)
        depths_arr[0] = 1
        return depths_arr
