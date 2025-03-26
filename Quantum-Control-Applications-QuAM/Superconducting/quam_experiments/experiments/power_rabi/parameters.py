from typing import Optional, Literal
import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a qubit spectroscopy experiment.

    Attributes:
        num_averages (int): Number of averages to perform. Default is 100.
        frequency_span_in_mhz (float): Span of frequencies to sweep in MHz. Default is 100 MHz.
        frequency_step_in_mhz (float): Step size for frequency sweep in MHz. Default is 0.25 MHz.
        operation (str): Type of operation to perform. Default is "saturation".
        operation_amplitude_prefactor (Optional[float]): Amplitude pre-factor for the operation. Default is 1.0.
        operation_len_in_ns (Optional[int]): Length of the operation in nanoseconds. Default is None.
        target_peak_width (Optional[float]): Target peak width in Hz. Default is 3e6 Hz.
    """

    num_averages: int = 50
    operation: Literal["x180", "x90", "-x90", "y90", "-y90"] = "x180"
    min_amp_factor: float = 0.001
    max_amp_factor: float = 1.99
    amp_factor_step: float = 0.005
    max_number_rabi_pulses_per_sweep: int = 1
    update_x90: bool = True


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass


def get_number_of_pulses(node_parameter: Parameters):
    if node_parameter.max_number_rabi_pulses_per_sweep > 1:
        if node_parameter.operation == "x180":
            N_pulses = np.arange(1, node_parameter.max_number_rabi_pulses_per_sweep, 2).astype("int")
        elif node_parameter.operation in ["x90", "-x90", "y90", "-y90"]:
            N_pulses = np.arange(2, node_parameter.max_number_rabi_pulses_per_sweep, 4).astype("int")
        else:
            raise ValueError(f"Unrecognized operation {node_parameter.operation}.")
    else:
        N_pulses = np.linspace(
            1,
            node_parameter.max_number_rabi_pulses_per_sweep,
            node_parameter.max_number_rabi_pulses_per_sweep,
        ).astype("int")[::2]
    return N_pulses
