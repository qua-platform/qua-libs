from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)

class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a resonator vs flux experiment.

    Attributes:
        num_averages (int): Number of averages to perform. Default is 100.
        min_flux_offset_in_v (float): Minimum flux bias offset in volts. Default is -0.5 V.
        max_flux_offset_in_v (float): Maximum flux bias offset in volts. Default is 0.5 V.
        num_flux_points (int): Number of flux points. Default is 101.
        frequency_span_in_mhz (float): Frequency span in MHz. Default is 15 MHz.
        frequency_step_in_mhz (float): Frequency step in MHz. Default is 0.1 MHz.
        input_line_impedance_in_ohm (float): Input line impedance in ohms. Default is 50 Ohm.
        line_attenuation_in_db (float): Line attenuation in dB. Default is 0 dB.
        update_flux_min (bool): Flag to update flux minimum frequency point. Default is False.
    """

    num_averages: int = 50
    operation: str = "saturation"
    operation_amplitude_factor: float = 0.1
    operation_len_in_ns: Optional[int] = None
    frequency_span_in_mhz: float = 100
    frequency_step_in_mhz: float = 0.1
    min_flux_offset_in_v: float = -0.02
    max_flux_offset_in_v: float = 0.03
    num_flux_points: int = 51


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
