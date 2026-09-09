"""Parameters module for N-qubit readout confusion matrix calibration."""

from typing import ClassVar, List, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from qualibration_libs.parameters.experiment import BaseExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for N-qubit confusion matrix measurement."""

    qubit_groups: Optional[List[List[str]]] = None
    """List of qubit groups; each group is a list of qubit names measured together."""

    num_shots: int = 2000
    """Number of shots per prepared state. Default is 2000."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    BaseExperimentNodeParameters,
    NodeSpecificParameters,
):
    """Combined parameters for N-qubit confusion matrix calibration."""

    targets_name: ClassVar[str] = "qubit_groups"
