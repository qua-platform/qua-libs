"""Parameters module for two-qubit readout confusion matrix calibration."""

from typing import ClassVar

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    CommonNodeParameters,
    QubitPairExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for two-qubit confusion matrix measurement."""

    num_shots: int = 2000
    """Number of shots per prepared state. Default is 2000."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for two-qubit confusion matrix calibration."""

    targets_name: ClassVar[str] = "qubit_pairs"
