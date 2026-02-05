from typing import Literal, Protocol, runtime_checkable

import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class BaseRabiSpecificParameters(RunnableParameters):
    """Parameters shared by nodes 08a (Power Rabi), 08b (Error Amplified Power Rabi), and 08c (Error Amplified Power Rabi Overtime)."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    min_amp_factor: float = 0.001
    """Minimum amplitude factor for the operation. Default is 0.001."""
    max_amp_factor: float = 1.99
    """Maximum amplitude factor for the operation. Default is 1.99."""
    amp_factor_step: float = 0.005
    """Step size for the amplitude factor. Default is 0.005."""
    operation: Literal["x180", "x90", "-x90", "y90", "-y90"] = "x180"
    """Type of operation to perform. Default is "x180"."""
    update_x90: bool = True
    """Flag to update the x90 pulse amplitude after calibrating x180. Default is True."""


class ErrorAmplifiedSpecificParameters(BaseRabiSpecificParameters):
    n_pulses: int = 1
    """Number of pulses in the error-amplified power Rabi pulse sequence."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    BaseRabiSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 08a_power_rabi."""


class ErrorAmplifiedParameters(
    NodeParameters,
    CommonNodeParameters,
    ErrorAmplifiedSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 08b_power_rabi_error_amplification"""
