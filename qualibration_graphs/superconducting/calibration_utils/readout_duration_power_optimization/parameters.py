"""Parameter definitions for the joint readout duration x power optimization."""

from typing import Literal

import numpy as np
from pydantic import model_validator

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters

# ``demod.accumulated`` expresses its chunk size in units of 4 ADC samples, so every chunk
# duration -- and therefore every integration duration on the sweep axis -- must land on a
# 4 ns grid.
NS_PER_CHUNK_UNIT = 4


class NodeSpecificParameters(RunnableParameters):
    """Sweep and analysis knobs for the joint duration / amplitude readout optimization."""

    num_shots: int = 2000
    """Number of shots per (amplitude, duration) point. Default is 2000."""
    start_amp: float = 0.5
    """First readout amplitude prefactor, relative to the current amplitude. Default is 0.5."""
    end_amp: float = 1.99
    """Last readout amplitude prefactor, relative to the current amplitude. Default is 1.99."""
    num_amps: int = 10
    """Number of amplitude prefactors to sweep. Default is 10."""
    max_duration_in_ns: int = 2000
    """Longest integration duration, and the readout pulse length used for the sweep. Default is 2000 ns."""
    num_durations: int = 10
    """Number of integration durations, evenly spaced up to the maximum. Default is 10."""
    outliers_threshold: float = 0.98
    """Minimum non-outlier fraction for a grid point to be eligible. Default is 0.98."""
    max_variance_ratio: float = 3.0
    """Maximum ratio between the wider and narrower fitted blob variance. Default is 3."""
    max_readout_amplitude: float = 0.125
    """Readout amplitude above which the node warns (it never clamps). Default is 0.125 V."""
    update_readout_length: bool = True
    """Whether to write the chosen integration duration to the readout pulse length. Default is True.

    When False the readout keeps the length it already has, and the operating-point search is
    pinned to that length so that the thresholds and the confusion matrix written to the state
    describe the integration duration the readout will actually run at. The node then behaves
    as an amplitude sweep, and a qubit whose current readout length is not on the swept
    duration axis fails with a note saying so."""
    operation: Literal["readout", "readout_QND"] = "readout"
    """Name of the resonator operation to optimize. Default is 'readout'."""

    @model_validator(mode="after")
    def check_chunk_duration_is_on_the_4ns_grid(self):
        """Accumulated demodulation tiles the pulse with equal chunks of a whole 4 ns unit."""
        if self.num_durations < 1:
            raise ValueError(f"num_durations must be >= 1, got {self.num_durations}.")
        chunk_ns = self.max_duration_in_ns / self.num_durations
        if chunk_ns % NS_PER_CHUNK_UNIT != 0:
            raise ValueError(
                f"max_duration_in_ns / num_durations must be a multiple of {NS_PER_CHUNK_UNIT} ns, got "
                f"{self.max_duration_in_ns} / {self.num_durations} = {chunk_ns} ns. "
                f"Try num_durations={self.max_duration_in_ns // 200} for a 200 ns chunk."
            )
        return self


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Aggregate node parameters for the joint readout duration x power optimization."""

    pass


def get_amplitude_prefactors(node_parameters: NodeSpecificParameters) -> np.ndarray:
    """The readout amplitude prefactors swept, relative to each qubit's current amplitude."""
    return np.linspace(node_parameters.start_amp, node_parameters.end_amp, node_parameters.num_amps)


def get_durations_in_ns(node_parameters: NodeSpecificParameters) -> np.ndarray:
    """The integration durations swept, evenly spaced from one chunk up to the maximum."""
    chunk_ns = get_chunk_duration_in_ns(node_parameters)
    return np.arange(1, node_parameters.num_durations + 1) * chunk_ns


def get_chunk_duration_in_ns(node_parameters: NodeSpecificParameters) -> int:
    """Duration of one accumulated-demodulation chunk, in nanoseconds."""
    return int(node_parameters.max_duration_in_ns // node_parameters.num_durations)


def get_samples_per_chunk(node_parameters: NodeSpecificParameters) -> int:
    """Chunk size in the units ``demod.accumulated`` expects, i.e. 4 ns each."""
    return get_chunk_duration_in_ns(node_parameters) // NS_PER_CHUNK_UNIT
