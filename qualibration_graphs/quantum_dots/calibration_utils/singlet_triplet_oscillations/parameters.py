from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import HeraldedInitializeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    """Number of averages to perform."""

    qubit_pair: str | None = None
    """Single qubit-pair name to run. If None, use the first active pair."""

    min_wait_duration_in_ns: int = 16
    """Minimum wait duration after initialise (ns, must be multiple of 4)."""

    max_wait_duration_in_ns: int = 4000
    """Maximum wait duration after initialise (ns, must be multiple of 4)."""

    wait_duration_step_in_ns: int = 20
    """Step size of the wait-duration sweep (ns, must be multiple of 4)."""

    analysis_signal: str = "state"
    """Signal to fit: 'state' or 'I'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    """Parameter set for 07e_singlet_triplet_oscillations."""
