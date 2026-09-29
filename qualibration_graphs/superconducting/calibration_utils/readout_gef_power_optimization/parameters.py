"""Parameter definitions for GEF readout power optimization."""

from typing import Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """GEF readout power optimization parameters for the amplitude sweep."""

    num_shots: int = 2000
    """Number of shots to perform. Default is 2000."""
    start_amp: float = 0.5
    """Start amplitude prefactor. Default is 0.5."""
    end_amp: float = 1.99
    """End amplitude prefactor. Default is 1.99."""
    num_amps: int = 10
    """Number of amplitudes to sweep. Default is 10."""
    operation: Literal["readout", "readout_QND", "readout_GEF"] = "readout_GEF"
    """Resonator operation to optimize. Default is 'readout_GEF'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for the GEF readout power optimization node."""

    pass
