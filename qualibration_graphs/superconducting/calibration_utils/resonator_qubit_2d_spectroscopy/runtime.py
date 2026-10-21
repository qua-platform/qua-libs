import logging
from typing import List, Optional

from quam_builder.architecture.superconducting.qubit import AnyTransmon

__all__ = [
    "estimate_reset_duration_in_ns",
    "estimate_sequence_duration_in_ns",
    "estimate_runtime_in_seconds",
    "log_runtime_estimate",
]


def estimate_reset_duration_in_ns(qubit: AnyTransmon, reset_type: str) -> float:
    """
    Returns the approximate duration of the qubit reset, in ns.

    A thermal reset is exactly `thermalization_time`. An active reset is data dependent, since it repeats the
    measure-and-conditional-pi cycle until the qubit is found in the ground state, so a single cycle is used here
    and the result is a lower bound.
    """
    if reset_type == "thermal":
        return float(qubit.thermalization_time)
    return float(
        qubit.resonator.operations["readout"].length
        + qubit.resonator.depletion_time / 2
        + qubit.xy.operations["x180"].length
        + 2000  # the fixed 500 clock cycle wait at the end of reset_qubit_active
    )


def estimate_sequence_duration_in_ns(qubit: AnyTransmon, parameters, stark_drive_len: int) -> float:
    """Returns the approximate duration of one grid point of the sequence for a single qubit, in ns."""
    duration = estimate_reset_duration_in_ns(qubit, parameters.reset_type)
    if "e" in parameters.qubit_initial_states:
        # Averaged over the interleaved states, only the ones prepared in "e" carry the extra pi pulse
        duration += qubit.xy.operations["x180"].length / len(parameters.qubit_initial_states)
    # The resonator drive covers the qubit probe, then the resonator is depleted before being read out
    duration += stark_drive_len
    duration += qubit.resonator.depletion_time
    duration += qubit.resonator.operations["readout"].length
    return float(duration)


def estimate_runtime_in_seconds(qubits: List[AnyTransmon], parameters, num_points: int, stark_drive_len: int) -> float:
    """
    Returns the approximate QUA runtime of the whole node, in seconds.

    Multiplexed qubits are reset, driven and measured together, so the slowest one sets the pace. Otherwise the
    sequence is replayed for each qubit in turn and the durations add up.

    Parameters
    ----------
    qubits : list of AnyTransmon
        The qubits taking part in the node.
    parameters
        The node parameters, used for `reset_type`, `qubit_initial_state`, `multiplexed` and `num_shots`.
    num_points : int
        The number of points of the 2D grid.
    stark_drive_len : int
        The duration of the resonator drive, in ns.

    Returns
    -------
    float
        The approximate runtime in seconds.
    """
    durations = [estimate_sequence_duration_in_ns(q, parameters, stark_drive_len) for q in qubits]
    duration_per_point = max(durations) if parameters.multiplexed else sum(durations)
    # Each grid point is replayed once per prepared state, interleaved within each shot
    return duration_per_point * num_points * parameters.num_shots * len(parameters.qubit_initial_states) * 1e-9


def log_runtime_estimate(
    qubits: List[AnyTransmon],
    parameters,
    num_points: int,
    stark_drive_len: int,
    log_callable: Optional[callable] = None,
) -> float:
    """
    Logs the approximate QUA runtime of the node, together with the knobs that shorten it.

    The 2D grid makes this node long: every one of its points carries a full qubit reset, which with a thermal
    reset is by far the dominant cost. Running against a backend that caps the execution time is the usual reason
    for a failure here, so the estimate is reported before the job is submitted.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    runtime = estimate_runtime_in_seconds(qubits, parameters, num_points, stark_drive_len)
    message = (
        f"Estimated QUA runtime: {runtime:.0f} s for {len(qubits)} qubit(s), {num_points} grid points, "
        f"{parameters.num_shots} shots and the state(s) {parameters.qubit_initial_states}, "
        f"with a '{parameters.reset_type}' reset "
        f"({'multiplexed' if parameters.multiplexed else 'sequential'})."
    )
    if parameters.reset_type == "thermal" or not parameters.multiplexed:
        message += (
            " Set 'multiplexed' to True and/or 'reset_type' to 'active' to shorten it, or reduce the number of "
            "qubits, the sweep density or 'num_shots'."
        )
    log_callable(message)
    return runtime
