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
    """Number of averages to perform. Default is 100."""
    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """Type of CZ operation to perform."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination for readout. Default is True."""
    max_circuit_depth: int = 64
    """Maximum circuit depth (number of Cliffords). Default is 64."""
    num_intervals: int = 7
    """Number of depth points from 0 to ``max_circuit_depth``. Default is 7."""
    interval_spacing: Literal["linear", "logarithmic"] = "logarithmic"
    """Depth spacing: ``linear`` uses ``np.linspace``; ``logarithmic`` uses
    ``np.geomspace(1, max_circuit_depth + 1, num_intervals) - 1``. Default is logarithmic."""
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
    max_chunk_ints: int = 15000
    """Maximum number of ints per input-stream chunk. Only used when
    use_input_stream=True. Must be < 16000 (the OPX QUA variable budget cap),
    with some headroom for the program's other declared variables. Default 15000."""
    verbose_memory_log: bool = False
    """Always logs per-depth transpile stats during encoding. When True, also
    logs RB circuit memory summary plus per-depth int and input-stream sub-chunk
    breakdown at QUA compile time."""
    reset_type: Literal["active", "thermal"] = "active"
    """Type of reset to perform. Default is active."""
    simulate: bool = False
    """Simulate the waveforms on the OPX instead of executing the program. Default is False."""
    fidelity_threshold: Optional[float] = None
    """Optional gate-fidelity acceptance threshold in [0, 1]. If set, qubit pairs whose fitted
    fidelity is below this value are additionally marked as failed in node.outcomes (in
    addition to fits that fail outright). For the standard RB node this is the 2Q Clifford
    fidelity; for the interleaved RB node it is the CZ gate fidelity. Used by higher-level
    adaptive graphs to route low-fidelity pairs to a retune subgraph via
    `connect_on_failure`. Default is None (no threshold check; only fit-failure marks a pair
    as failed)."""
    rb_plot_style: Literal["error_bars", "per_sequence"] = "error_bars"
    """How to display RB survival data in ``plot_data``.
    - ``error_bars`` (default): one point per circuit depth showing the mean
      P(|00>) averaged over all shots and random sequences, with vertical
      error bars giving the standard error of that mean (SEM).
    - ``per_sequence``: at each depth, plot one point per random RB sequence
      (shot-averaged P(|00>) for that circuit) as a light scatter cloud, plus
      the depth mean on top. Shows circuit-to-circuit spread rather than
      uncertainty on the averaged mean. No error bars in this mode.
    """
    rb_plot_log_x: bool = False
    """If True, use a logarithmic x-axis (circuit depth) in ``plot_data``.
    Useful when depths span several orders of magnitude. Depth 0 (if present)
    is omitted from the log scale."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for two-qubit randomized benchmarking experiments."""

    targets_name: ClassVar[str] = "qubit_pairs"


# Named dimension order after fetch (must match QuaProgramHandler stream buffers).
STREAMED_RAW_DIMS = ("qubit_pair", "circuit_depth", "sequence", "shots")
DECLARED_RAW_DIMS = ("qubit_pair", "shots", "circuit_depth", "sequence")
CANONICAL_ANALYSIS_DIMS = ("qubit_pair", "shots", "circuit_depth", "sequence")


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

    **Without input stream** the program loops multiplex → shot → circuit
    (depth-major) → gate. Stream buffers are
    ``.buffer(sequence).buffer(circuit_depth).buffer(shots)``, so raw axes are
    ``qubit_pair, shots, circuit_depth, sequence``.

    **With input stream** the program loops multiplex → depth/chunk → circuit
    → shot → gate, with buffers
    ``.buffer(num_shots).buffer(num_circuits_per_depth).buffer(num_depths)``.
    Raw axes are ``qubit_pair, circuit_depth, sequence, shots``.

    ``process_raw_dataset`` transposes streamed data to the canonical named
    layout ``qubit_pair, shots, circuit_depth, sequence``.
    """
    if use_input_stream:
        return {
            "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
            "circuit_depth": xr.DataArray(np.array(circuit_depths)),
            "sequence": xr.DataArray(np.arange(num_circuits_per_depth)),
            "shots": xr.DataArray(np.arange(num_shots)),
        }
    return {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "shots": xr.DataArray(np.arange(num_shots)),
        "circuit_depth": xr.DataArray(np.array(circuit_depths)),
        "sequence": xr.DataArray(np.arange(num_circuits_per_depth)),
    }


def rb_progress_total(
    num_shots: int,
    num_circuits_per_depth: int,
    num_depths: int,
    *,
    use_input_stream: bool,
) -> int:
    """Denominator for ``progress_counter`` (completed circuit repetitions).

    Streamed mode saves progress at every circuit×shot boundary. Declared-array
    mode still saves once per outer shot.
    """
    if use_input_stream:
        return num_shots * num_circuits_per_depth * num_depths
    return num_shots
