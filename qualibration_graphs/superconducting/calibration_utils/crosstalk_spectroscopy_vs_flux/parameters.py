from typing import Optional, Literal

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    operation: str = "saturation"
    """Operation to perform. Default is "saturation"."""
    operation_amplitude_factor: float = 0.1
    """Amplitude factor for the operation. Default is 0.1."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in ns. Default is the predefined pulse length."""
    frequency_span_in_mhz: float = 100.0
    """Frequency span in MHz. Default is 100 MHz."""
    frequency_num_points: float = 51
    """Frequency step in MHz. Default is 0.1 MHz."""
    flux_span_in_v: float = 0.2
    """Minimum flux bias offset in volts. Default is -0.02 V."""
    flux_num_points: int = 51
    """Number of flux points. Default is 51."""
    flux_detuning_mode: Literal["auto_for_linear_response", "auto_fill_sweep_window", "manual"] = "auto_for_linear_response"
    """Strategy for choosing the target qubit's flux detuning."""
    manual_flux_detuning_in_v: float = None
    """Target qubit's flux detuning when the mode is set to manual."""
    expected_crosstalk: float = 0.01
    """Change in target qubit flux per unit of aggressor qubit flux. """
    flux_pulse_padding_in_ns: float = 2000
    """Extra padding time between the flux pulse and pi-pulse, which is also doubled and added to the duration of the flux pulse"""
    input_line_impedance_in_ohm: Optional[int] = 50
    """Input line impedance in ohms. Default is 50 Ohm."""
    line_attenuation_in_db: Optional[int] = 0
    """Line attenuation in dB. Default is 0 dB."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
