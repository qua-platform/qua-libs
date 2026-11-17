"""Parameter definitions for the G-E-F readout duration optimization (node 14c)."""

from typing import Literal

from pydantic import model_validator

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters

# The duration axis is the accumulated-demodulation chunk grid of node 08c, so the helpers that
# build it are shared rather than duplicated.
from calibration_utils.readout_duration_power_optimization.parameters import (  # noqa: F401
    NS_PER_CHUNK_UNIT,
    get_chunk_duration_in_ns,
    get_durations_in_ns,
    get_samples_per_chunk,
)


class NodeSpecificParameters(RunnableParameters):
    """Sweep and selection knobs for the G-E-F readout duration optimization."""

    num_shots: int = 2000
    """Number of single shots per prepared state. Every shot yields every integration duration.
    Default is 2000."""
    max_duration_in_ns: int = 3000
    """Longest integration duration, and the readout pulse length played during the sweep.
    Default is 3000 ns."""
    num_durations: int = 15
    """Number of integration durations, evenly spaced up to the maximum. Default is 15."""
    fidelity_tolerance: float = 0.005
    """The chosen duration is the shortest one whose three-state assignment fidelity is within this
    absolute tolerance of the best fidelity on the axis. Default is 0.005."""
    min_assignment_fidelity: float = 0.5
    """Minimum three-state assignment fidelity (mean of the confusion-matrix diagonal) for the fit to
    succeed. Random assignment gives 1/3. Default is 0.5."""
    operation: Literal["readout_GEF", "readout", "readout_QND"] = "readout_GEF"
    """Resonator operation whose length is optimized. Default is 'readout_GEF'."""

    @model_validator(mode="after")
    def check_chunk_duration_is_on_the_4ns_grid(self):
        """Accumulated demodulation tiles the pulse with equal chunks of a whole 4 ns unit."""
        if self.num_durations < 1:
            raise ValueError(f"num_durations must be >= 1, got {self.num_durations}.")
        chunk_ns = self.max_duration_in_ns / self.num_durations
        if chunk_ns % NS_PER_CHUNK_UNIT != 0:
            raise ValueError(
                f"max_duration_in_ns / num_durations must be a multiple of {NS_PER_CHUNK_UNIT} ns, got "
                f"{self.max_duration_in_ns} / {self.num_durations} = {chunk_ns} ns."
            )
        if not 0 <= self.fidelity_tolerance < 1:
            raise ValueError(f"fidelity_tolerance must be in [0, 1), got {self.fidelity_tolerance}.")
        return self


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Aggregate node parameters for the G-E-F readout duration optimization."""

    pass
