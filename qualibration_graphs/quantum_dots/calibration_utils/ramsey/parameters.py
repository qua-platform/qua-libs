from typing import Literal, Protocol, runtime_checkable

import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Parameters shared by both 04b (GE power Rabi) and 12b (EF power Rabi) nodes."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_detuning_in_mhz: float = 1.0
    """Frequency detuning in MHz. Default is 1.0 MHz."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 10a_ramsey_parity_diff."""
    pass
