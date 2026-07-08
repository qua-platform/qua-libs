"""Parameter definitions for two-qubit randomized benchmarking experiments.

This module defines the parameters used for configuring RB experiments,
including circuit lengths, number of shots, and operation types.
"""

# pylint: disable=duplicate-code,too-few-public-methods

from typing import ClassVar, Literal, Optional

import numpy as np
import xarray as xr
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for two-qubit RB experiments."""

    num_shots: int = 100
    """Number of averages to perform. Default is 50."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """Type of CZ operation to perform."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""
    circuit_depths: list[int] = [1, 2, 4, 8, 16, 32, 64]
    """Circuit lengths (number of Cliffords) to benchmark. Default is (1, 4, 16, 32, 64)."""
    num_circuits_per_depth: int = 5
    """Number of random circuits sampled per circuit length. Default is 5."""
    seed: int = 0
    """Random seed for circuit generation to ensure reproducibility. Default is 0."""
    use_input_stream: bool = False
    """Whether to use input streams for circuit execution. Default is False.
    When True, the gate sequences are streamed to the OPX chunk-by-chunk via the
    QUA input-stream feature instead of being declared as a single large
    `declare(int, value=...)` array. This bypasses the OPX's ~16000 QUA variable
    budget cap on declared arrays, enabling longer circuit depths and/or more
    circuits per depth than the without-input-stream path can support."""
    max_chunk_ints: int = 16000
    """Maximum number of ints per input-stream chunk. Only used when
    use_input_stream=True. Must be < 16000 (the OPX QUA variable budget cap),
    with some headroom for the program's other declared variables. Default 15500."""
    verbose_memory_log: bool = False
    """Always logs per-depth transpile stats during encoding. When True, also
    logs RB circuit memory summary plus per-depth int and input-stream sub-chunk
    breakdown at QUA compile time."""
    reset_type: Literal["active", "thermal"] = "active"
    """Type of reset to perform. Default is active."""
    fidelity_threshold: Optional[float] = None
    """Optional gate-fidelity acceptance threshold in [0, 1]. If set, qubit pairs whose fitted
    fidelity is below this value are additionally marked as failed in node.outcomes (in
    addition to fits that fail outright). For the standard RB node this is the 2Q Clifford
    fidelity; for the interleaved RB node it is the CZ gate fidelity. Used by higher-level
    adaptive graphs to route low-fidelity pairs to a retune subgraph via
    `connect_on_failure`. Default is None (no threshold check; only fit-failure marks a pair
    as failed)."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for two-qubit randomized benchmarking experiments."""

    targets_name: ClassVar[str] = "qubit_pairs"


def build_sweep_axes(
    qubit_pairs,
    num_shots: int,
    circuit_depths: list[int],
    num_circuits_per_depth: int,
    *,
    use_input_stream: bool,
) -> dict[str, xr.DataArray]:
    """Build sweep axes for :class:`~qualibration_libs.data.XarrayDataFetcher`.

    The dict **key order** must match the dimension order of the raw arrays
    produced by QUA stream processing. ``XarrayDataFetcher`` assigns fetched
    data to coordinates in insertion order; a mismatch raises incompatible-shape
    errors at fetch time.

    **Without input stream** the QUA program loops shots on the outside and
    plays the full flattened circuit list each shot. Stream buffers are
    ``.buffer(sequence).buffer(circuit_depth).buffer(shots)``, so measurement
    order is ``(shots, circuit_depth, sequence)`` and the axes are ordered
    ``qubit_pair, shots, circuit_depth, sequence``.

    **With input stream** the program advances one host-pushed sub-chunk at a
    time and replays all shots on the OPX before the next advance (one push per
    sub-chunk instead of per shot). That changes the save order to
    ``(circuit_depth, shots, sequence)``, with buffers
    ``.buffer(sequence).buffer(shots).buffer(circuit_depth)`` and axes
    ``qubit_pair, circuit_depth, shots, sequence``.

    Analysis still expects canonical ``(shots, circuit_depth, sequence)``
    layout; ``process_raw_dataset`` transposes input-stream datasets after fetch.
    """
    if use_input_stream:
        return {
            "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
            "circuit_depth": xr.DataArray(np.array(circuit_depths)),
            "shots": xr.DataArray(np.arange(num_shots)),
            "sequence": xr.DataArray(np.arange(num_circuits_per_depth)),
        }
    return {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "shots": xr.DataArray(np.arange(num_shots)),
        "circuit_depth": xr.DataArray(np.array(circuit_depths)),
        "sequence": xr.DataArray(np.arange(num_circuits_per_depth)),
    }
