"""Parameter definitions for single-qubit gate set tomography experiments.

This module defines the parameters used for configuring GST experiments,
including circuit lengths, number of shots, and operation types.
"""

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for single-qubit GST experiments."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    germ_lengths: list[int] = [1, 4, 16, 32, 64]
    """Number of times a germ is repeated in the sequence. Default is [1, 4, 16, 32, 64]."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""
    use_strict_timing: bool = False
    """Use strict timing in the QUA program. Default is False."""
    use_input_stream: bool = False
    """Whether to use input streams for circuit execution. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for single-qubit GST experiments."""
