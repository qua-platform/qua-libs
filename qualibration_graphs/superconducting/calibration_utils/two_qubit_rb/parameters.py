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
    """Stream the gate sequences to the OPX in chunks instead of one declared array.
    Bypasses the ~16000 QUA variable cap, allowing deeper or more circuits. Default is False."""
    max_chunk_ints: int = 15000
    """Maximum ints per input-stream chunk (only with ``use_input_stream``).
    Must stay below 16000 with headroom for other variables. Default is 15000."""
    verbose_memory_log: bool = False
    """Also log the circuit memory summary and per-depth chunk breakdown at compile time.
    Per-depth transpile stats are always logged. Default is False."""
    reset_type: Literal["active", "thermal"] = "active"
    """Type of reset to perform. Default is active."""
    measure_leakage: bool = False
    """Read both qubits as g/e/f and fit leakage; needs 15_iq_blobs_gef and a thermal reset.
    Writes only ``LeakageRB_reference`` (37a) / ``LeakageRB`` (37b) to the state. Default is False."""
    simulate: bool = False
    """Simulate the waveforms on the OPX instead of executing the program. Default is False."""
    fidelity_threshold: Optional[float] = None
    """Mark pairs below this fidelity as failed (2Q Clifford fidelity in 37a, CZ fidelity in 37b).
    Lets adaptive graphs route them to a retune. Default is None (only fit failures)."""
    rb_plot_style: Literal["error_bars", "per_sequence"] = "error_bars"
    """``error_bars``: depth mean with its SEM. ``per_sequence``: one point per random
    sequence plus the depth mean, no error bars. Default is error_bars."""
    rb_plot_log_x: bool = False
    """Use a logarithmic circuit-depth axis in ``plot_data`` (depth 0 is omitted). Default is False."""

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


def read_confusion_matrix(qubit) -> Optional[np.ndarray]:
    """The qubit's g/e/f confusion matrix (rows prepared), or ``None`` if missing or not a valid 3x3."""
    matrix = getattr(qubit.resonator, "gef_confusion_matrix", None)
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.allclose(matrix.sum(axis=1), 1.0, atol=0.02):
        return None
    return matrix


def validate_leakage_run(node, qubit_pairs) -> None:
    """Check a ``measure_leakage`` run can start, and log qubits read without confusion correction.

    Raises:
        ValueError: If the reset is not thermal, or a qubit has no ``readout_GEF`` operation or no
            three calibrated ``gef_centers`` (node 15_iq_blobs_gef).
    """
    if node.parameters.reset_type != "thermal":
        raise ValueError(
            "measure_leakage needs reset_type='thermal': the g/e active reset cannot return |f> "
            "to |g>, so leaked population would carry over into the next shot."
        )
    qubits = {qubit.name: qubit for qp in qubit_pairs for qubit in (qp.qubit_control, qp.qubit_target)}
    problems = []
    for name, qubit in qubits.items():
        if "readout_GEF" not in qubit.resonator.operations:
            problems.append(f"{name}: no 'readout_GEF' operation")
        centers = getattr(qubit.resonator, "gef_centers", None) or []
        if len(centers) != 3 or any(len(center) != 2 for center in centers):
            problems.append(f"{name}: no three calibrated gef_centers")
    if problems:
        raise ValueError(
            "measure_leakage needs a calibrated g/e/f readout (run 15_iq_blobs_gef first): " + "; ".join(problems)
        )
    uncorrected = sorted(name for name, qubit in qubits.items() if read_confusion_matrix(qubit) is None)
    if uncorrected:
        node.log(
            f"No valid gef_confusion_matrix for {', '.join(uncorrected)}: their populations are not "
            "corrected for readout misassignment. The fitted L1 is a slope, so a constant "
            "misassignment mostly cancels, but treat these with care.",
            level="warning",
        )
