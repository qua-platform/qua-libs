"""Parameter definitions for Ramsey versus coupler flux calibration."""

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for Ramsey vs coupler flux experiment.

    This experiment performs a Ramsey sequence on one qubit of a pair while
    sweeping the flux on the coupler, characterising how the coupler flux
    affects the measured qubit frequency.
    """

    num_shots: int = 100
    """Number of averages per point."""
    measure_qubit: Literal["control", "target"] = "control"
    """Which qubit in the pair to perform the Ramsey measurement on."""
    frequency_detuning_in_mhz: float = 5.0
    """Artificial detuning for the virtual Z rotation (MHz)."""
    coupler_flux_min: float = -0.05
    """Minimum coupler flux (V)."""
    coupler_flux_max: float = 0.05
    """Maximum coupler flux (V)."""
    coupler_flux_num: int = 101
    """Number of coupler flux points to sample."""
    min_wait_time_in_ns: int = 16
    """Minimum Ramsey idle time (ns). Must be >= 16."""
    max_wait_time_in_ns: int = 1000
    """Maximum Ramsey idle time (ns)."""
    wait_time_step_in_ns: int = 4
    """Step size for the idle time sweep (ns)."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for the Ramsey vs coupler flux calibration node."""

    targets_name: ClassVar[str] = "qubit_pairs"
