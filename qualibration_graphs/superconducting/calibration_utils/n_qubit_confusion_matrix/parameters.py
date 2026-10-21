"""Parameters module for N-qubit readout confusion matrix calibration."""

from typing import ClassVar, List, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from qualibration_libs.parameters.experiment import BaseExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for N-qubit confusion matrix measurement."""

    qubit_groups: Optional[List[str]] = None
    """Qubit groups to measure, one entry per group.

    Each entry is a dash-separated list of qubit names measured together, e.g.
    ``["qC4-qC3-qC2", "qC1-qC2"]``.
    """

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
