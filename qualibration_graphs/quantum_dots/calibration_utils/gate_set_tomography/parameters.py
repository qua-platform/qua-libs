"""Parameter definitions for single-qubit gate set tomography experiments.

This module defines the parameters used for configuring GST experiments,
including circuit lengths, number of shots, and operation types.
"""

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for single-qubit GST experiments."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""
    max_germ_repetition: int = 64
    """Maximum number of times a germ is repeated in the sequence. Default is 64."""
    use_input_stream: bool = False
    """Whether to use input streams for circuit execution. Default is False."""
    reset_type: Literal["active", "thermal"] = "active"


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for single-qubit GST experiments."""
