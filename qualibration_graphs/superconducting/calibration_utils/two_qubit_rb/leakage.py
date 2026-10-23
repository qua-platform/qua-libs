"""Leakage analysis for the two-qubit RB nodes run with ``measure_leakage=True``.

Each shot's state is ``3 * control + target`` with each qubit read as g (0), e (1) or f (2).
Per qubit pair this module fits the leaked population (either qubit in f) and each qubit's own
f population against circuit depth (see :mod:`calibration_utils.common_utils.leakage_rb`).

37a (reference) fits leakage per 2Q Clifford and stores it as ``LeakageRB_reference``. 37b
(CZ interleaved after every Clifford) subtracts that reference to give leakage per CZ, stored
as ``LeakageRB``. Both live in ``qubit_pair.macros[operation].fidelity``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from qualibrate import QualibrationNode
from qualibration_libs.plotting import grid_iter

from calibration_utils.common_utils.leakage_rb import (
    LeakageFit,
    check_gef_readout,
    fit_leakage_decay,
    interleaved_gate_leakage,
    mean_over_sequences,
    plot_leakage_decay,
    read_confusion_matrix,
    two_qubit_leakage,
)
from calibration_utils.pair_grid import QubitPairGrid, grid_pair_names

REFERENCE_KEY = "LeakageRB_reference"
"""Fidelity-dict key 37a writes: leakage per 2Q Clifford."""

INTERLEAVED_KEY = "LeakageRB"
"""Fidelity-dict key 37b writes: leakage per CZ."""

SERIES = ("pair", "control", "target")
"""Leaked populations fitted per qubit pair: either qubit in f, control in f, target in f."""


def validate_leakage_run(node: QualibrationNode, qubit_pairs) -> None:
    """Check a ``measure_leakage`` run can start, and log pairs read without correction.

    Raises:
        ValueError: If the reset is not thermal or a qubit has no g/e/f readout.
    """
    if node.parameters.reset_type != "thermal":
        raise ValueError(
            "measure_leakage needs reset_type='thermal': the g/e active reset cannot return |f> "
            "to |g>, so leaked population would carry over into the next shot."
        )
    qubits = {}
    for qp in qubit_pairs:
        for qubit in (qp.qubit_control, qp.qubit_target):
            qubits[qubit.name] = qubit
    check_gef_readout(qubits.values())
    uncorrected = [name for name, qubit in qubits.items() if read_confusion_matrix(qubit) is None]
    if uncorrected:
        node.log(
            "No valid gef_confusion_matrix for "
            + ", ".join(sorted(uncorrected))
            + ": their populations are not corrected for readout misassignment. The fitted L1 "
            "is a slope, so a constant misassignment mostly cancels, but treat these with care.",
            level="warning",
        )


def analyse_leakage(
    ds: xr.Dataset, node: QualibrationNode, *, interleaved: bool
) -> tuple[xr.Dataset, Dict[str, Dict[str, Any]]]:
    """Fit leakage for every qubit pair.

    Args:
        ds: Processed raw dataset with ``state`` over ``qubit_pair, shots, circuit_depth, sequence``.
        node: The running node (qubit pairs, operation, machine for the reference).
        interleaved: Whether the circuits interleave the CZ (37b); the result then also holds the
            per-CZ leakage against the stored reference.

    Returns:
        A dataset of leaked populations and fitted curves per pair and depth, and a JSON-safe
        result dict per pair.
    """
    qubit_pairs = {qp.name: qp for qp in node.namespace["qubit_pairs"]}
    depths = np.asarray(ds.circuit_depth.values, dtype=float)
    pair_names = [str(name) for name in ds.qubit_pair.values]

    data: dict[str, np.ndarray] = {}
    for series in SERIES:
        for suffix in ("", "_stderr", "_fit"):
            data[f"leaked_{series}{suffix}"] = np.full((len(pair_names), len(depths)), np.nan)

    results: Dict[str, Dict[str, Any]] = {}
    for p, qp_name in enumerate(pair_names):
        qp = qubit_pairs[qp_name]
        states = ds.state.sel(qubit_pair=qp_name).transpose("circuit_depth", "sequence", "shots").values
        confusion_control = read_confusion_matrix(qp.qubit_control)
        confusion_target = read_confusion_matrix(qp.qubit_target)
        leaked = two_qubit_leakage(states, -1, confusion_control, confusion_target)

        fits = {}
        for series in SERIES:
            mean, stderr = mean_over_sequences(leaked[series], sequence_axis=-1)
            fit = fit_leakage_decay(depths, mean, stderr)
            fits[series] = fit
            data[f"leaked_{series}"][p] = mean
            data[f"leaked_{series}_stderr"][p] = stderr
            data[f"leaked_{series}_fit"][p] = fit.curve(depths)

        result: Dict[str, Any] = {
            "control_qubit": qp.qubit_control.name,
            "target_qubit": qp.qubit_target.name,
            "confusion_corrected": {
                "control": confusion_control is not None,
                "target": confusion_target is not None,
            },
            "per_clifford": {series: fits[series].to_jsonable() for series in SERIES},
            "success": fits["pair"].success,
        }
        if interleaved:
            result.update(_per_cz(node, qp_name, fits))
        results[qp_name] = result

    coords = {"qubit_pair": pair_names, "circuit_depth": depths}
    ds_leak = xr.Dataset(
        {name: (("qubit_pair", "circuit_depth"), values) for name, values in data.items()},
        coords=coords,
    )
    return ds_leak, results


def _per_cz(node: QualibrationNode, qp_name: str, fits: dict) -> Dict[str, Any]:
    """Per-CZ leakage of each series against the reference 37a stored for this pair."""
    fidelity = node.machine.qubit_pairs[qp_name].macros[node.parameters.operation].fidelity
    reference = fidelity.get(REFERENCE_KEY)
    if not reference:
        return {
            "per_cz": None,
            "reference_load_id": None,
            "per_cz_note": (
                f"No {REFERENCE_KEY} for {qp_name}: run 37a_two_qubit_standard_rb with "
                "measure_leakage=True first. Only the leakage per interleaved Clifford is reported."
            ),
        }
    per_cz = {}
    for series in SERIES:
        ref = (reference.get("per_clifford") or {}).get(series) or {}
        per_cz[series] = interleaved_gate_leakage(
            fits[series],
            ref.get("l1"),
            ref.get("l1_stderr"),
            ref.get("l2"),
            ref.get("l2_stderr"),
        ).to_jsonable()
    return {"per_cz": per_cz, "reference_load_id": reference.get("load_id")}


def state_entry(result: Dict[str, Any], load_id: Optional[int], *, interleaved: bool) -> Dict[str, Any]:
    """The dict a pair's result writes to ``fidelity[REFERENCE_KEY]`` or ``fidelity[INTERLEAVED_KEY]``."""
    entry = {
        "control_qubit": result["control_qubit"],
        "target_qubit": result["target_qubit"],
        "confusion_corrected": result["confusion_corrected"],
        "per_clifford": result["per_clifford"],
        "load_id": load_id,
    }
    if interleaved:
        entry["per_cz"] = result.get("per_cz")
        entry["reference_load_id"] = result.get("reference_load_id")
    return entry


def _format_rate(fit: Optional[Dict[str, Any]]) -> str:
    if not fit or fit.get("l1") is None:
        return "n/a"
    stderr = fit.get("l1_stderr")
    return f"{fit['l1']:.2e}" + (f" +/- {stderr:.1e}" if stderr is not None else "")


def log_leakage_results(
    results: Dict[str, Dict[str, Any]], *, interleaved: bool, log_callable: Callable = print
) -> None:
    """Log L1 per pair: per Clifford, and per CZ for the interleaved node."""
    for qp_name, result in results.items():
        per_clifford = result["per_clifford"]
        unit = "interleaved Clifford" if interleaved else "2Q Clifford"
        lines = [f"Qubit pair {qp_name} leakage L1 per {unit}:"]
        for series in SERIES:
            label = {"pair": "either qubit", "control": result["control_qubit"], "target": result["target_qubit"]}
            lines.append(f"  {label[series]}: {_format_rate(per_clifford[series])}")
            note = per_clifford[series].get("note")
            if note:
                lines.append(f"    ({note})")
        if interleaved:
            per_cz = result.get("per_cz")
            if per_cz is None:
                lines.append(f"  per CZ: {result.get('per_cz_note')}")
            else:
                lines.append(
                    f"  per CZ: either qubit {_format_rate(per_cz['pair'])}, "
                    f"{result['control_qubit']} {_format_rate(per_cz['control'])}, "
                    f"{result['target_qubit']} {_format_rate(per_cz['target'])}"
                )
        log_callable("\n".join(lines))


def plot_leakage(
    node: QualibrationNode, ds_leak: xr.Dataset, results: Dict[str, Dict[str, Any]], *, title: str
) -> Dict[str, Figure]:
    """Plot leaked population against depth, one panel per qubit pair on the chip grid."""
    grid_names, pair_names = grid_pair_names(node.namespace["qubit_pairs"])
    grid = QubitPairGrid(grid_names, pair_names)
    depths = ds_leak.circuit_depth.values
    interleaved = "interleaved" in node.name.lower()

    for ax, qubit in grid_iter(grid):
        qp_name = qubit["qubit"]
        if qp_name not in results:
            ax.text(0.5, 0.5, "No leakage data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Qubit pair: {qp_name}")
            continue
        result = results[qp_name]
        labels = {"pair": "either qubit", "control": result["control_qubit"], "target": result["target_qubit"]}
        fr = ds_leak.sel(qubit_pair=qp_name)
        for series in SERIES:
            stored = result["per_clifford"][series]
            fit = LeakageFit(**{k: (np.nan if v is None else v) for k, v in stored.items()})
            plot_leakage_decay(
                ax,
                depths,
                fr[f"leaked_{series}"].values,
                fr[f"leaked_{series}_stderr"].values,
                fit,
                label=f"{labels[series]}: L1 {_format_rate(result['per_clifford'][series])}",
            )
        subtitle = ""
        if interleaved and result.get("per_cz"):
            subtitle = f"\nper CZ: {_format_rate(result['per_cz']['pair'])}"
        ax.set_title(f"{qp_name}{subtitle}", fontsize=9)
        ax.set_xlabel("Circuit depth (Cliffords)")
        ax.set_ylabel("Leaked population")
        ax.legend(fontsize=7)

    grid.fig.suptitle(f"{title}\n({node.parameters.reset_type} reset, g/e/f readout)")
    grid.fig.tight_layout()
    return {"leakage": grid.fig}
