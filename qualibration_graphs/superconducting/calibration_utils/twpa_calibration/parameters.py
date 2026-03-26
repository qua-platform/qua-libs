from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    twpa: str = None
    """TWPA to calibrate. Must be a string. Default is None"""
    num_shots: int = 1000
    """Number of averages to perform. Default is 30."""
    max_power_dbm_p: int = 2
    """Maximum twpa pump power level in dBm. Default is -25 dBm."""
    min_power_dbm_p: int = -2
    """Minimum twpa pump power level in dBm. Default is -50 dBm."""
    num_power_points_p: int = 20
    """Number of points of the twpa pump power axis. Default is 100."""
    max_amp_p: float = 0.8
    """Maximum twpa pump amplitude for the experiment. Default is 0.1."""
    frequency_span_in_mhz_p: float = 60
    """Span of twpa pump frequencies to sweep in MHz. Default is 60 MHz."""
    frequency_step_in_mhz_p: float = 1
    """Step size for twpa pump frequency sweep in MHz. Default is 0.5 MHz."""
    optimizer_method: str = "average"
    """Method to get the best pump parameters. Can be either "average" (get the best SNR averaged across all qubits), or "worst-qubit" (get the best SNR for the worst qubit). Default is "average"."""
    min_gain: float = 0.0
    """Minimum gain allowed for optimizing the pump parameters based on the SNR. Default is 0."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
