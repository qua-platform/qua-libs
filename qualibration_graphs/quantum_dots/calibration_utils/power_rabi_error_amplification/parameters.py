from typing import Literal
import numpy as np

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters
from calibration_utils.heralded_initialization_utils import HeraldedInitializeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    min_amp_factor: float = 0.85
    """Minimum amplitude prefactor. Narrow window around expected a_π after node 09a."""
    max_amp_factor: float = 1.15
    """Maximum amplitude prefactor. Narrow window around expected a_π after node 09a."""
    amp_factor_step: float = 0.001
    """Step size for the amplitude prefactor sweep. Default is 0.001."""
    max_n_pulses: int = 40
    """Number of pulses in the error-amplified power Rabi pulse sequence."""
    operation: Literal["x180", "x90"] = "x180"
    """The operation to perform to drive the qubit."""
    use_simulated_data: bool = False
    """Whether to generate simulated data instead of measuring via the OPX. Default False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 09b_power_rabi_error_amplification."""


def get_number_of_pulses(node_parameter: NodeSpecificParameters):
    """Return array of number of pulses for error amplification."""

    if node_parameter.max_n_pulses > 1:
        if node_parameter.operation == "x180":
            N_pulses = np.arange(1, node_parameter.max_n_pulses, 2).astype(int)
        elif node_parameter.operation in ["x90", "-x90", "y90", "-y90"]:
            N_pulses = np.arange(2, node_parameter.max_n_pulses, 4).astype(int)
        else:
            raise ValueError(f"Unrecognized operation {node_parameter.operation}.")
    else:
        N_pulses = np.linspace(
            1,
            node_parameter.max_n_pulses,
            node_parameter.max_n_pulses,
        ).astype(
            int
        )[::2]
    return N_pulses
