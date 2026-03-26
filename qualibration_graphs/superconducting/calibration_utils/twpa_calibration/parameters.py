from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    twpas: list[str] = ["twpaA"]
    """List of twpas to calibrate"""
    num_shots: int = 30
    """Number of averages to perform. Default is 30."""
    frequency_span_in_mhz: float = 4
    """Span of readout frequencies to sweep in MHz. Default is 4 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for readout frequency sweep in MHz. Default is 0.1 MHz."""
    max_power_dbm_p: int = 3
    """Maximum twpa pump power level in dBm. Default is -25 dBm."""
    min_power_dbm_p: int = -10
    """Minimum twpa pump power level in dBm. Default is -50 dBm."""
    num_power_points_p: int = 10
    """Number of points of the twpa pump power axis. Default is 100."""
    max_amp_p: float = 1
    """Maximum twpa pump amplitude for the experiment. Default is 0.1."""
    frequency_span_in_mhz_p: float = 60
    """Span of twpa pump frequencies to sweep in MHz. Default is 60 MHz."""
    frequency_step_in_mhz_p: float = 0.5
    """Step size for twpa pump frequency sweep in MHz. Default is 0.5 MHz."""

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
