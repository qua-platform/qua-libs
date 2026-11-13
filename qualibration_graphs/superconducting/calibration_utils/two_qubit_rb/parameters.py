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

from calibration_utils.common_utils.rb import validate_leakage_run as validate_common_leakage_run


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
    """Stream the gate sequences to the OPX in chunks instead of one declared array.
    Bypasses the ~16000 QUA variable cap, allowing deeper or more circuits. Default is False."""
    max_chunk_ints: int = 15000
    """Maximum ints per input-stream chunk (only with ``use_input_stream``).
    Must stay below 16000 with headroom for other variables. Default is 15000."""
    verbose_memory_log: bool = False
    """Also log the circuit memory summary and per-depth chunk breakdown at compile time.
    Per-depth transpile stats are always logged. Default is False."""
    reset_type: Literal["active", "thermal", "active_gef"] = "active"
    """Type of reset to perform. Default is active."""
    measure_leakage: bool = False
    """Read both qubits as g/e/f and fit leakage; needs 15_iq_blobs_gef and a thermal or active_gef reset.
    Writes ``macros[operation].extras["leakage_per_2q_clifford"]`` (37a) / ``["leakage_per_cz"]`` (37b)
    to the state, next to the usual RB fidelity. Default is False."""
    simulate: bool = False
    """Simulate the waveforms on the OPX instead of executing the program. Default is False."""
    fidelity_threshold: Optional[float] = None
    """Mark pairs below this fidelity as failed (2Q Clifford fidelity in 37a, CZ fidelity in 37b).
    Lets adaptive graphs route them to a retune. Default is None (only fit failures)."""
    rb_plot_style: Literal["error_bars", "per_sequence"] = "error_bars"
    """``error_bars``: depth mean with its SEM. ``per_sequence``: one point per random
    sequence plus the depth mean, no error bars. Default is error_bars."""
    rb_plot_log_x: bool = False
    """Use a logarithmic circuit-depth axis in ``plot_data`` (depth 0 is omitted). Also applies to
    the leakage plot. Default is False."""


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


def validate_leakage_run(node, qubit_pairs) -> None:
    """Run the shared leakage pre-run check on every control and target qubit, each once."""
    qubits = {qubit.name: qubit for qp in qubit_pairs for qubit in (qp.qubit_control, qp.qubit_target)}
    validate_common_leakage_run(node, list(qubits.values()))


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
