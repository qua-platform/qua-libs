"""Parameter definitions for IQ blobs GEF calibration experiment."""

from typing import Literal
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """IQ blobs GEF specific parameters for three-state measurement."""

    num_shots: int = 2000
    """Number of runs to perform. Default is 2000."""
    operation: Literal["readout", "readout_QND", "readout_GEF"] = "readout_GEF"
    """Type of operation to perform. Default is "readout_GEF"."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for IQ blobs GEF calibration node."""

    pass
