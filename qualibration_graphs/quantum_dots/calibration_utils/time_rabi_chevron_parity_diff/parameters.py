"""Node parameters for time Rabi chevron parity difference calibration."""

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    min_wait_time_in_ns: int = 16
    """Minimum pulse duration in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns."""
    max_wait_time_in_ns: int = 10_000
    """Maximum pulse duration in nanoseconds. Default is 10000 ns (10 us)."""
    time_step_in_ns: int = 52
    """Step size for the pulse duration sweep in nanoseconds. Default is 52 ns."""
    frequency_span_in_mhz: float = 2
    """Span of frequencies to sweep in MHz. Default is 2 MHz."""
    frequency_step_in_mhz: float = 0.025
    """Step size for the frequency detuning sweep in MHz. Default is 0.025 MHz."""
    gap_wait_time_in_ns: int = 128
    """Wait time between initialization and X180 pulse in nanoseconds. Default is 128 ns."""
    operation: str = "x180"
    """Name of the qubit operation to perform. Default is 'x180'."""
    use_numpyro: bool = True
    """If True, use Bayesian (NumPyro NUTS) fit instead of scipy curve_fit. Default is True."""
    mcmc_num_warmup: int = 500
    """Number of NUTS warmup iterations. Default is 500."""
    mcmc_num_samples: int = 500
    """Number of posterior samples per chain. Default is 500."""
    mcmc_num_chains: int = 1
    """Number of MCMC chains. Default is 1."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
