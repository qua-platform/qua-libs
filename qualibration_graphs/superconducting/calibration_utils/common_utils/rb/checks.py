"""Pre-run checks for leakage RB, shared by the single- and two-qubit RB nodes."""

from __future__ import annotations

from .populations import read_confusion_matrix


def validate_leakage_run(node, qubits) -> None:
    """Check a ``measure_leakage`` run can start, and log qubits read without confusion correction.

    Args:
        node: The running node; ``node.parameters.reset_type`` is checked and warnings go to ``node.log``.
        qubits: The qubits that are read out, each listed once.

    Raises:
        ValueError: If the reset is neither thermal nor active_gef, or a qubit has no ``readout_GEF`` or
            (for active_gef) ``EF_x180`` operation, or no three calibrated ``gef_centers``
            (node 15_iq_blobs_gef).
    """
    if node.parameters.reset_type not in ("thermal", "active_gef"):
        raise ValueError(
            "measure_leakage needs reset_type='thermal' or 'active_gef': the g/e active reset cannot "
            "return |f> to |g>, so leaked population would carry over into the next shot."
        )
    problems = []
    for qubit in qubits:
        if "readout_GEF" not in qubit.resonator.operations:
            problems.append(f"{qubit.name}: no 'readout_GEF' operation")
        centers = getattr(qubit.resonator, "gef_centers", None) or []
        if len(centers) != 3 or any(len(center) != 2 for center in centers):
            problems.append(f"{qubit.name}: no three calibrated gef_centers")
        if node.parameters.reset_type == "active_gef" and "EF_x180" not in qubit.xy.operations:
            problems.append(f"{qubit.name}: no 'EF_x180' operation (the active_gef reset plays it)")
    if problems:
        raise ValueError(
            "measure_leakage needs a calibrated g/e/f readout (run 15_iq_blobs_gef first): " + "; ".join(problems)
        )
    uncorrected = sorted(qubit.name for qubit in qubits if read_confusion_matrix(qubit) is None)
    if uncorrected:
        node.log(
            f"No valid gef_confusion_matrix for {', '.join(uncorrected)}: their populations are not "
            "corrected for readout misassignment. The fitted L1 is a slope, so a constant "
            "misassignment mostly cancels, but treat these with care.",
            level="warning",
        )
