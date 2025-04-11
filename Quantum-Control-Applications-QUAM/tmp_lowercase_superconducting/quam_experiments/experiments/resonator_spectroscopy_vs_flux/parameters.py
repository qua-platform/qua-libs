from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_averages: int = 100
    """Number of averages to perform. Default is 100."""
    min_flux_offset_in_v: float = -0.5
    """Minimum flux bias offset in volts. Default is -0.5 V."""
    max_flux_offset_in_v: float = 0.5
    """Maximum flux bias offset in volts. Default is 0.5 V."""
    num_flux_points: int = 101
    """Number of flux points. Default is 101."""
    frequency_span_in_mhz: float = 15
    """Frequency span in MHz. Default is 15 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Frequency step in MHz. Default is 0.1 MHz."""
    input_line_impedance_in_ohm: float = 50
    """Input line impedance in ohms. Default is 50 Ohm."""
    line_attenuation_in_db: float = 0
    """Line attenuation in dB. Default is 0 dB."""
    update_flux_min: bool = False
    """Flag to update flux minimum frequency point. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
