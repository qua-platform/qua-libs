from typing import List, Literal, Optional, get_args

import numpy as np
from pydantic import Field
from qualibrate import NodeParameters
from qualibrate.core.parameters import GroupParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters

from .dd_sequences import DD_SEQUENCES, MIN_WAIT_CC, DDSequence, get_dd_sequence

# Patches grouped-parameter defaults before this node is constructed, so the GUI keeps submitted values.
import qualibrate_group_defaults  # noqa: F401, E402


class DDSequenceParameters(GroupParameters):
    """DD sequence and idle window."""

    sequence: Literal["CPMG", "XY4", "XY8", "XY16"] = "CPMG"
    """DD sequence; block sizes 2, 4, 8, 16 pulses. Default is "CPMG"."""
    window_ns: Optional[int] = None
    """Idle window in ns. If None, the longest readout pulse of the selected qubits. Default is None."""
    pulses_per_window: Optional[List[int]] = None
    """Pi pulses per window to scan (multiples of the block size). If None, all up to the 32 ns minimum spacing."""
    use_strict_timing: bool = True
    """Raise an error instead of silently adding gaps between pulses. Default is True."""


class DDSweepParameters(GroupParameters):
    """Sampling of the decay curves."""

    num_shots: int = 400
    """Number of averages. Default is 400."""
    max_num_windows: int = 60
    """Longest decay curve, in windows (aim for ~3x T2). Default is 60."""
    num_time_points: int = 40
    """Points per decay curve between 0 and max_num_windows windows. Default is 40."""


class DDDecisionParameters(GroupParameters):
    """Choice of the optimal number of pulses."""

    num_rounds: int = 10
    """Windows M over which the error per round is averaged, from the data after M windows. Default is 10."""
    max_extra_error_per_round: float = 1e-3
    """Accepted extra error per round over the best N; the optimum is the fewest pulses within it. Default is 1e-3."""
    uncertainty_margin_sigma: float = 1.0
    """Noise allowance in standard errors before an N is rejected. Default is 1.0."""


class DDVisualizationParameters(GroupParameters):
    """Optional figures."""

    show_noise_spectrum: bool = True
    """Fit and plot the first-order dephasing noise spectrum. Default is True."""


assert set(get_args(DDSequenceParameters.__annotations__["sequence"])) == set(
    DD_SEQUENCES
), "The `sequence` Literal must list the same names as DD_SEQUENCES"


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
):

    sequence: DDSequenceParameters = Field(default_factory=DDSequenceParameters)
    sweep: DDSweepParameters = Field(default_factory=DDSweepParameters)
    decision: DDDecisionParameters = Field(default_factory=DDDecisionParameters)
    visualization: DDVisualizationParameters = Field(default_factory=DDVisualizationParameters)


def get_window_ns(node_parameters: Parameters, qubits) -> int:
    """Return the idle window duration in ns, rounded down to a multiple of 4 ns."""
    if node_parameters.sequence.window_ns is not None:
        window = node_parameters.sequence.window_ns
    else:
        window = max(q.resonator.operations["readout"].length for q in qubits)
    return int(window) // 4 * 4


def get_pi_length_ns(qubit, sequence: DDSequence) -> int:
    """Longest pi pulse used by the sequence for this qubit (they are normally all the same length)."""
    missing = [op for op in sequence.operations if op not in qubit.xy.operations]
    if missing:
        raise ValueError(f"{qubit.name}: operations {missing} needed by the {sequence.name} sequence are not defined.")
    return max(int(qubit.xy.operations[op].length) for op in sequence.operations)


def max_pulses_per_window(window_ns: int, pi_length_ns: int, pulses_per_block: int = 2) -> int:
    """Largest multiple of the block size N such that pulses are >= 32 ns apart: N * (32 ns + t_pi) <= window."""
    n_max = window_ns // (2 * 4 * MIN_WAIT_CC + pi_length_ns)
    return int(n_max - n_max % pulses_per_block)


def get_pulses_per_window(
    node_parameters: Parameters, window_ns: int, pi_lengths_ns, pulses_per_block: int = 2
) -> np.ndarray:
    """Return the numbers of pi pulses per window to scan (multiples of the block size), limited by the minimum
    spacing for the longest pi pulse."""
    n_max = max_pulses_per_window(window_ns, max(pi_lengths_ns), pulses_per_block)
    if n_max < pulses_per_block:
        raise ValueError(
            f"The window ({window_ns} ns) is too short to fit one {pulses_per_block}-pulse block with the minimum "
            "spacing."
        )
    if node_parameters.sequence.pulses_per_window is None:
        return np.arange(pulses_per_block, n_max + 1, pulses_per_block)
    n_values = np.array(sorted(set(node_parameters.sequence.pulses_per_window)), dtype=int)
    if np.any(n_values % pulses_per_block) or np.any(n_values < pulses_per_block):
        raise ValueError(f"pulses_per_window must be positive multiples of {pulses_per_block}, got {list(n_values)}.")
    if np.any(n_values > n_max):
        raise ValueError(f"pulses_per_window must be <= {n_max} for a {window_ns} ns window, got {list(n_values)}.")
    return n_values


def get_window_counts(max_num_windows: int, num_points: int, num_rounds: int = 1) -> np.ndarray:
    """Number of windows for each point of a decay curve: distinct integers spread linearly over 0..max_num_windows,
    always including 1 window and num_rounds windows.

    Shared by all qubits and numbers of pulses, so every decay curve has the same time axis (windows * window_ns).
    """
    if not 1 <= num_rounds <= max_num_windows:
        raise ValueError(f"num_rounds must be between 1 and max_num_windows ({max_num_windows}), got {num_rounds}.")
    counts = np.round(np.linspace(0, max_num_windows, num_points)).astype(int)
    return np.unique(np.concatenate([counts, [1, num_rounds]]))


def get_sweep_schedule(node_parameters: Parameters, qubits) -> dict:
    """Compute the full DD sweep: window, pulses per window, window counts per point and per-qubit free evolution time
    per window (in clock cycles)."""
    sequence = get_dd_sequence(node_parameters.sequence.sequence)
    window_ns = get_window_ns(node_parameters, qubits)
    pi_lengths = {q.name: get_pi_length_ns(q, sequence) for q in qubits}
    n_values = get_pulses_per_window(node_parameters, window_ns, list(pi_lengths.values()), sequence.pulses_per_block)
    free_cc = {name: np.array([(window_ns - n * t_pi) // 4 for n in n_values]) for name, t_pi in pi_lengths.items()}
    return {
        "sequence": sequence.name,
        "window_ns": window_ns,
        "pulses_per_window": n_values,
        "window_counts": get_window_counts(
            node_parameters.sweep.max_num_windows,
            node_parameters.sweep.num_time_points,
            node_parameters.decision.num_rounds,
        ),
        "pi_lengths": pi_lengths,
        "free_cc": free_cc,
    }


def assign_schedule_coords(ds, schedule: dict):
    """Attach the DD timing coordinates (times in ns) to a dataset with dims (qubit, pulses_per_window, point)."""
    names = [str(q) for q in ds.qubit.values]
    n_values = np.asarray(schedule["pulses_per_window"])
    windows = np.asarray(schedule["window_counts"])
    free_ns = np.array([4 * schedule["free_cc"][q] for q in names])
    spacing_ns = np.broadcast_to(schedule["window_ns"] / n_values, free_ns.shape).copy()
    return ds.assign_coords(
        windows=(("point",), windows, {"long_name": "number of windows"}),
        time=(("point",), windows * schedule["window_ns"], {"long_name": "total evolution time", "units": "ns"}),
        total_pulses=(("pulses_per_window", "point"), n_values[:, None] * windows[None, :]),
        tau=(
            ("qubit", "pulses_per_window"),
            free_ns / (2 * n_values),
            {"long_name": "pulse half-spacing", "units": "ns"},
        ),
        pulse_spacing=(("qubit", "pulses_per_window"), spacing_ns, {"long_name": "pi-pulse spacing", "units": "ns"}),
        window_ns=schedule["window_ns"],
        sequence=schedule["sequence"],
    )
