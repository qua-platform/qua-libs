"""Data fetching and pyGSTi analysis for gate set tomography (AIS)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pygsti
import xarray as xr
from qualibrate import QualibrationNode
from qualibrate.core.utils.node.path_solver import get_node_dir_path
from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path

from .gst_utils import GSTExperimentDesign


def shots_to_count_dataset(flat, qubit_name: str, total_germs_num: int, n_runs: int) -> xr.Dataset:
    """Integer counts from a shot record, or rint reconstruction from per-circuit averages.

    A full shot record (n_circuits * n_runs values) is summed, so n1 is exact.
    An already-averaged record (one value per circuit) uses np.rint(state * n_runs):
    astype(int) truncates 29/100*100 to 28.
    """
    flat = np.asarray(flat, dtype=float).ravel()
    germs = np.arange(total_germs_num)
    if flat.size == total_germs_num * n_runs:
        shots = flat.reshape(total_germs_num, n_runs).astype(np.int64)
        count1 = shots.sum(axis=1)
        state = count1 / n_runs
    elif flat.size == total_germs_num:
        state = flat
        count1 = np.rint(flat * n_runs).astype(np.int64)
    elif flat.size == n_runs:
        raise ValueError(
            f"Got {n_runs} values (one per run, not per circuit). The compiled QUA "
            "program likely still has runs as the outer loop. Re-run the QUA program "
            "cell so the loop order is: outer germs, inner runs."
        )
    else:
        raise ValueError(
            f"Expected shape ({total_germs_num}, {n_runs}) from "
            f"buffer({n_runs}).buffer({total_germs_num}), got {flat.size} values. "
            "Re-run the QUA program cell after code changes."
        )
    count0 = np.int64(n_runs) - count1
    return xr.Dataset(
        {
            "state": (("qubit", "germs"), state.reshape(1, -1)),
            "count1": (("qubit", "germs"), count1.reshape(1, -1)),
            "count0": (("qubit", "germs"), count0.reshape(1, -1)),
        },
        coords={"qubit": [qubit_name], "germs": germs},
    )


def recompute_counts_from_state(ds: xr.Dataset, n_runs: int) -> xr.Dataset:
    """Rebuild count0/count1 from the saved state fraction.

    state = k/n is exact for the shot counts this node stores, and np.rint recovers k.
    Datasets saved before the count fix stored count1 = astype(int)(state * 100), which
    is wrong whenever num_shots != 100 and is biased low at k = 29, 57, 58 for n = 100.
    """
    count1 = np.rint(np.asarray(ds.state.values, dtype=float) * n_runs).astype(np.int64)
    ds = ds.copy()
    ds["count1"] = (("qubit", "germs"), count1)
    ds["count0"] = (("qubit", "germs"), np.int64(n_runs) - count1)
    return ds


def fetch_gst_state_averaged(handles, qubit, total_germs_num: int, n_runs: int) -> xr.Dataset:
    """Fetch 2D results (germs x runs) and reduce to exact integer counts per circuit."""
    flat = np.array(handles.get("state1").fetch_all(), dtype=float).ravel()
    return shots_to_count_dataset(flat, qubit.name, total_germs_num, n_runs)


def build_raw_dataset(handles, qubits, design: GSTExperimentDesign, n_runs: int) -> xr.Dataset:
    """Fetch and combine per-circuit counts for all qubits."""
    ds = None
    for qubit in qubits:
        ds_ = fetch_gst_state_averaged(handles, qubit, design.total_germs_num, n_runs)
        ds = xr.concat([ds, ds_], dim="qubit") if ds is not None else ds_
    if not np.all(ds.count0.values + ds.count1.values == n_runs):
        raise ValueError(f"count0 + count1 is not {n_runs} for every circuit.")
    return ds


def transform_dataset_to_gst(ds: xr.Dataset, design: GSTExperimentDesign, qubit_index: int = 0) -> pygsti.data.DataSet:
    """Convert an xarray dataset into a pyGSTi DataSet."""
    gst_ds = pygsti.data.DataSet(outcome_labels=["0", "1"])
    for i, crc in enumerate(design.exp_design.all_circuits_needing_data):
        gst_ds.add_count_dict(
            crc,
            {"0": int(ds.count0.values[qubit_index, i]), "1": int(ds.count1.values[qubit_index, i])},
        )
    return gst_ds


def _pp_vector_to_stdmx(obj) -> np.ndarray:
    """Convert a pyGSTi state/effect object or vector to a standard density matrix."""
    if isinstance(obj, np.ndarray):
        vec = obj
    else:
        # TPState.to_vector() omits the fixed trace-normalization element (len=3 for 1Q),
        # so use the full dense representation instead.
        vec = obj.to_dense(on_space="minimal")
    return pygsti.tools.vec_to_stdmx(vec, basis="pp")


def _empty_gst_results_template() -> dict[str, dict[str, Any]]:
    """Return the results dictionary skeleton for one qubit."""
    empty_op = {"choi": None, "fidelity": None, "robustness": None}
    return {
        "TP": {
            "rho0": {"density_mx": None, "fidelity": None},
            "meas_op": {
                "0": {"povm": None, "fidelity": None},
                "1": {"povm": None, "fidelity": None},
            },
            "gate_op": {"I": dict(empty_op), "x90": dict(empty_op), "y90": dict(empty_op)},
        },
        "CPTP": {
            "rho0": {"density_mx": None, "fidelity": None},
            "meas_op": {
                "0": {"povm": None, "fidelity": None},
                "1": {"povm": None, "fidelity": None},
            },
            "gate_op": {"I": dict(empty_op), "x90": dict(empty_op), "y90": dict(empty_op)},
        },
        "Ideal": {
            "rho0": {"density_mx": None, "fidelity": None},
            "meas_op": {
                "0": {"povm": None, "fidelity": None},
                "1": {"povm": None, "fidelity": None},
            },
            "gate_op": {"I": dict(empty_op), "x90": dict(empty_op), "y90": dict(empty_op)},
        },
    }


def run_gst_analysis(
    ds: xr.Dataset,
    design: GSTExperimentDesign,
    qubit_name: str,
    qubit_index: int = 0,
    log_callable: Callable = print,
) -> tuple[object, dict[str, dict[str, Any]]]:
    """Run pyGSTi StandardGST and extract fidelities for one qubit."""
    gst_ds = transform_dataset_to_gst(ds, design, qubit_index=qubit_index)
    gst_data = pygsti.protocols.ProtocolData(design.exp_design, gst_ds)
    gst_protocol = pygsti.protocols.StandardGST()
    gst_results = gst_protocol.run(gst_data, disable_checkpointing=True)

    gst_results_dict = _empty_gst_results_template()
    estimate_keys = ["full TP", "CPTPLND", "Target"]
    native_gate_keys = [(), ("Gxpi2", 0), ("Gypi2", 0)]
    std_model = design.std_model

    np.set_printoptions(suppress=True, precision=6)
    for i, cond in enumerate(gst_results_dict.keys()):
        est_model = gst_results.estimates[estimate_keys[i]].models["stdgaugeopt"]
        log_callable(f"\n=== Under {cond} condition ({qubit_name}) ===")

        rho_est_mat = _pp_vector_to_stdmx(est_model.preps["rho0"])
        rho_std_mat = _pp_vector_to_stdmx(std_model.preps["rho0"])
        state_fidelity = pygsti.tools.fidelity(rho_est_mat, rho_std_mat)
        log_callable(f"State preparation fidelity for '0' state is {state_fidelity:.6f}.")

        gst_results_dict[cond]["rho0"]["density_mx"] = rho_est_mat
        gst_results_dict[cond]["rho0"]["fidelity"] = state_fidelity

        povm_obj = est_model.povms["Mdefault"]
        meas_lines = []
        for label, effect_vec in povm_obj.items():
            matrix_est_form = _pp_vector_to_stdmx(effect_vec)
            matri_std_form = _pp_vector_to_stdmx(std_model.povms["Mdefault"][str(label)])
            meas_fidelity = pygsti.tools.fidelity(matrix_est_form, matri_std_form)
            meas_lines.append(f"'{label}' state is {meas_fidelity:.6f}")

            gst_results_dict[cond]["meas_op"][str(label)]["povm"] = matrix_est_form
            gst_results_dict[cond]["meas_op"][str(label)]["fidelity"] = meas_fidelity
        log_callable("State measurement fidelity for " + " ".join(meas_lines))

        for j, gate in enumerate(gst_results_dict[cond]["gate_op"].keys()):
            matrix_ptm = est_model.operations[native_gate_keys[j]].to_dense()
            choi = pygsti.tools.jamiolkowski.jamiolkowski_iso(matrix_ptm, op_mx_basis="pp", choi_mx_basis="std")
            infidelity = pygsti.tools.entanglement_infidelity(
                matrix_ptm,
                std_model.operations[native_gate_keys[j]].to_dense(),
                "pp",
            )
            gst_results_dict[cond]["gate_op"][gate]["choi"] = choi
            gst_results_dict[cond]["gate_op"][gate]["fidelity"] = 1 - infidelity

            log_callable(f"\nGate {gate} infidelity: {infidelity:.6f}")
            evals = np.linalg.eigvals(choi)
            log_callable(f"Choi eigenvalues: {np.round(evals.real, 6)}")

    return gst_results, gst_results_dict


def write_gst_html_report(
    gst_results,
    qubit_name: str,
    snapshot_idx: int,
    report_dirname: Optional[str] = None,
) -> tuple[str, str]:
    """Write the pyGSTi HTML report next to the saved node data."""
    report_dirname = report_dirname or f"gst_report_{qubit_name}"
    qs = get_qualibrate_config(get_qualibrate_config_path())
    node_dir = Path(get_node_dir_path(snapshot_idx, qs.storage.location))
    gst_report_dir = node_dir / report_dirname

    try:
        report = pygsti.report.construct_standard_report(
            gst_results,
            title=f"GST Report - {qubit_name}",
        )
        report.write_html(str(gst_report_dir), auto_open=False)
        main_html = gst_report_dir / "main.html"
        if not main_html.exists():
            raise RuntimeError("pyGSTi finished without creating main.html")
        print(f"GST HTML report saved to {main_html}")
    except Exception as exc:
        raise RuntimeError(
            "Failed to generate GST HTML report. "
            "pyGSTi requires plotly<6 for HTML reports; ensure a compatible plotly version is installed."
        ) from exc

    return report_dirname, f"{report_dirname}/main.html"


def analyse_gst_data(
    node: QualibrationNode,
    ds: xr.Dataset,
    design: GSTExperimentDesign,
) -> Dict[str, object]:
    """Run GST analysis for each qubit and store results on the node."""
    qubits = node.namespace["qubits"]
    node.results["gst_results"] = {}
    gst_result_objects = {}

    for q_idx, qubit in enumerate(qubits):
        gst_results, gst_results_dict = run_gst_analysis(
            ds,
            design,
            qubit.name,
            qubit_index=q_idx,
            log_callable=node.log,
        )
        node.results["gst_results"][qubit.name] = gst_results_dict
        gst_result_objects[qubit.name] = gst_results

    return gst_result_objects
