from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters
from typing import Literal

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    operation: Literal["x180", "x90", "-x90", "y90", "-y90"] = "x180"
    """Type of operation to perform. Default is "x180"."""
    min_amp_factor: float = 0.001
    """Minimum amplitude factor for the operation. Default is 0.001."""
    max_amp_factor: float = 1.99
    """Maximum amplitude factor for the operation. Default is 1.99."""
    amp_factor_step: float = 0.005
    """Step size for the amplitude factor. Default is 0.005."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
