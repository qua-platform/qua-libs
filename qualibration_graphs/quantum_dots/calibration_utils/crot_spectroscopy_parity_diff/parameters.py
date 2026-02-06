from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 2
    """Span of frequencies to sweep in MHz. Default is 2 MHz."""
    frequency_step_in_mhz: float = 0.025
    """Step size for the frequency detuning sweep in MHz. Default is 0.025 MHz."""
    min_voltage_in_v: float = 0.0
    """Minimum virtual barrier/exchange voltage in V. Default is 0.0 V."""
    max_voltage_in_v: float = 0.1
    """Maximum virtual barrier/exchange voltage in V. Default is 0.1 V."""
    voltage_step_in_v: float = 0.001
    """Step size for the virtual barrier/exchange voltage sweep in V. Default is 0.001 V."""
    duration_in_ns: int = 1000
    """Duration of the CROT drive pulse in nanoseconds. Default is 1000 ns."""
    operation: str = "x180"
    """Name of the qubit operation to perform. Default is 'x180'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    BaseExperimentNodeParameters,
    QubitPairExperimentNodeParameters,
):
    pass
