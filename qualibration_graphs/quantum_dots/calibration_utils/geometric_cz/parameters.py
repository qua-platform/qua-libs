from typing import List, Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    min_exchange_duration_in_ns: int = 16
    """Minimum exchange pulse duration in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns."""
    max_exchange_duration_in_ns: int = 2000
    """Maximum exchange pulse duration in nanoseconds. Default is 2000 ns (2 us)."""
    duration_step_in_ns: int = 20
    """Step size for the exchange pulse duration sweep in nanoseconds. Default is 20 ns."""
    min_exchange_amplitude: float = 0.0
    """Minimum exchange pulse amplitude (virtual barrier voltage). Default is 0.0 V."""
    max_exchange_amplitude: float = 0.5
    """Maximum exchange pulse amplitude (virtual barrier voltage). Default is 0.5 V."""
    amplitude_step: float = 0.01
    """Step size for the exchange pulse amplitude sweep in Volts. Default is 0.01 V."""
    target_qubit: str = "q1"
    """The target qubit whose phase will be measured. Default is 'q1'."""
    control_qubit: str = "q2"
    """The control qubit whose state conditions the phase. Default is 'q2'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    pass
