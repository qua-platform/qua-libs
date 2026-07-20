"""Parameters for All-XY calibration."""

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters

from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """All-XY-specific parameters."""

    num_shots: int = 500
    """Number of averages to perform. Default is 500."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for All-XY calibration node."""

    pass
