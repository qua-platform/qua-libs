from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 10.0
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 1.0
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    min_amp_factor: float = 0.2
    """Minimum amplitude factor for the operation. Default is 0.35."""
    max_amp_factor: float = 0.6
    """Maximum amplitude factor for the operation. Default is  0.7."""
    amp_factor_step: float = 0.05
    """Step size for the amplitude factor. Default is 0.01."""
    pump_frequency_span_in_mhz: float = 100.0
    """Span of TWPA pump frequencies to sweep in MHz. Default is 300 MHz."""
    pump_frequency_step_in_mhz: float = 1.0
    """Step size for TWPA pump frequency sweep in MHz. Default is 0.5 MHz."""
    pumpline_attenuation: int = -50 - 6  # (-50: fridge atten, directional coupler, -6: room temp line, -5: fridge line)


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
