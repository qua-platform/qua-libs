from typing import Optional

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 50
    """Number of averages to perform. Default is 50."""
    operation: str = "saturation"
    """Operation to perform. Default is "saturation"."""
    operation_amplitude_factor: float = 0.1
    """Amplitude factor for the operation. Default is 0.1."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in ns. Default is the predefined pulse length."""
    frequency_span_in_mhz: float = 100.0
    """Frequency span in MHz. Default is 100 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Frequency step in MHz. Default is 0.1 MHz."""
    min_flux_offset_in_v: float = -0.02
    """Minimum flux bias offset in volts. Default is -0.02 V."""
    max_flux_offset_in_v: float = 0.03
    """Maximum flux bias offset in volts. Default is 0.03 V."""
    num_flux_points: int = 51
    """Number of flux points. Default is 51."""
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
