from typing import Literal, Protocol, runtime_checkable

import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node specific parameters for 11_hahn_echo_parity_diff."""
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    tau_min: int = 16
    """Minimum pulse duration in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns."""
    tau_max: int = 10_000
    """Maximum pulse duration in nanoseconds. Default is 100000 ns (10 µs)."""
    tau_step: int = 16
    """Step size for the pulse duration sweep in nanoseconds. Default is 16 ns."""
    operation: str = "x180"
    """Name of the qubit operation to perform. Default is 'x180'."""

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 11_hahn_echo_parity_diff."""
