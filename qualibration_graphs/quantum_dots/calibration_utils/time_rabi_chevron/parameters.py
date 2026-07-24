"""Node parameters for time Rabi chevron calibration."""

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters
from calibration_utils.heralded_initialization_utils import HeraldedInitializeParameters
from calibration_utils.measurement_utils import ParityDiffAnalysisParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    min_wait_time_in_ns: int = 16
    """Minimum pulse duration in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns."""
    max_wait_time_in_ns: int = 10_000
    """Maximum pulse duration in nanoseconds. Default is 10000 ns (10 us)."""
    time_step_in_ns: int = 52
    """Step size for the pulse duration sweep in nanoseconds. Default is 52 ns."""
    frequency_span_in_mhz: float = 5
    """Span of frequencies to sweep in MHz. Default is 5 MHz."""
    frequency_step_in_mhz: float = 0.05
    """Step size for the frequency detuning sweep in MHz. Default is 0.05 MHz."""
    operation: str = "x180"
    """Name of the qubit operation to perform. Default is 'x180'."""
    parity_measurement: bool = False
    """Whether or not to perform parity measurement."""
    use_simulated_data: bool = False
    """Whether to generate simulated data instead of measuring via the OPX. Default False."""

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 10b_time_rabi_chevron."""
