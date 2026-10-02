"""Data fetching and pyGSTi analysis for two-qubit gate set tomography (AIS)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pygsti
import xarray as xr
from qualibrate import QualibrationNode
from qualibrate.core.utils.node.path_solver import get_node_dir_path
from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path

from .gst_utils import GST2QExperimentDesign, OUTCOME_LABELS

_OUTCOME_FROM_INT = {i: label for i, label in enumerate(OUTCOME_LABELS)}


def fetch_gst_2q_counts(handles, design: GST2QExperimentDesign, n_runs: int) -> xr.Dataset:
    """Histogram per-shot 2Q states (0..3) into 00/01/10/11 counts per circuit."""
    handle = handles.get("state2q")
    if handle is None:
        raise KeyError("Job has no 'state2q' result handle.")
    flat = np.array(handle.fetch_all(), dtype=int).ravel()
    n_circuits = design.total_germs_num
    expected = n_circuits * n_runs
    if flat.size != expected:
        raise ValueError(
            f"Expected shape ({n_circuits}, {n_runs}) from "
            f"buffer({n_runs}).buffer({n_circuits}), got {flat.size} values."
        )
    shots = flat.reshape(n_circuits, n_runs)
    counts = {label: np.zeros(n_circuits, dtype=int) for label in OUTCOME_LABELS}
    for outcome, label in _OUTCOME_FROM_INT.items():
        counts[label] = np.sum(shots == outcome, axis=1).astype(int)

    data_vars = {f"count_{label}": (("circuit",), counts[label]) for label in OUTCOME_LABELS}
    return xr.Dataset(
        data_vars,
        coords={"circuit": np.arange(n_circuits)},
    )


def build_raw_dataset_2q(handles, design: GST2QExperimentDesign, n_runs: int, pair_name: str) -> xr.Dataset:
    """Fetch 2Q GST counts and tag them with the qubit-pair name."""
    ds = fetch_gst_2q_counts(handles, design, n_runs)
    ds = ds.expand_dims(qubit_pair=[pair_name])
    return ds


def transform_dataset_to_gst_2q(ds: xr.Dataset, design: GST2QExperimentDesign) -> pygsti.data.DataSet:
    """Convert histogrammed 2Q counts into a pyGSTi DataSet."""
    if "qubit_pair" in ds.dims:
        ds = ds.isel(qubit_pair=0)

    gst_ds = pygsti.data.DataSet(outcome_labels=list(OUTCOME_LABELS))
    circuits = list(design.exp_design.all_circuits_needing_data)
    for i, crc in enumerate(circuits):
        count_dict = {label: int(ds[f"count_{label}"].values[i]) for label in OUTCOME_LABELS}
        try:
            gst_ds.add_count_dict(crc, count_dict)
        except Exception:
            gst_ds.add_count_dict(
                crc,
                {tuple(label): n for label, n in count_dict.items()},
            )
    if hasattr(gst_ds, "done_adding_data"):
        gst_ds.done_adding_data()
    return gst_ds


def _pp_vector_to_stdmx(obj) -> np.ndarray:
    """Convert a pyGSTi state/effect object or vector to a standard density matrix."""
    if isinstance(obj, np.ndarray):
        vec = obj
    else:
        vec = obj.to_dense(on_space="minimal")
    return pygsti.tools.vec_to_stdmx(vec, basis="pp")


def _dense_operation(op) -> np.ndarray:
    try:
        return np.asarray(op.to_dense(on_space="HilbertSchmidt"))
    except TypeError:
        return np.asarray(op.to_dense())


def _estimate_model(estimate):
    models = getattr(estimate, "models", {})
    for key in ("stdgaugeopt", "go0", "final iteration estimate", "iteration estimates"):
        if key in models:
            value = models[key]
            return value[-1] if isinstance(value, (list, tuple)) else value
    for value in models.values():
        if hasattr(value, "operations"):
            return value
        if isinstance(value, (list, tuple)) and value and hasattr(value[-1], "operations"):
            return value[-1]
    raise RuntimeError("Could not locate the gauge-optimised fitted model.")


def _call_metric(name: str, estimated, target) -> Optional[float]:
    tools = pygsti.tools
    fn = getattr(tools, name, None)
    if fn is None and hasattr(tools, "optools"):
        fn = getattr(tools.optools, name, None)
    if fn is None:
        return None
    for args in (
        (estimated, target),
        (_dense_operation(estimated), _dense_operation(target)),
    ):
        try:
            return float(np.real_if_close(fn(*args)))
        except Exception:
            continue
    return None


def _pretty_gate_name(label) -> str:
    text = str(label)
    mapping = {
        "()": "I",
        "Gxpi2:0": "x90_q0",
        "Gxpi2:1": "x90_q1",
        "Gypi2:0": "y90_q0",
        "Gypi2:1": "y90_q1",
        "Gcphase:0:1": "CZ",
        "Gcphase": "CZ",
    }
    for key, value in mapping.items():
        if key in text.replace(" ", ""):
            return value
    return text


def extract_gate_metrics(estimated_model, target_model) -> dict[str, dict[str, Any]]:
    """Pull infidelity metrics for every operation shared with the target model."""
    gate_metrics = {}
    for label, estimated_op in estimated_model.operations.items():
        if label not in target_model.operations:
            continue
        target_op = target_model.operations[label]
        infidelity = _call_metric("entanglement_infidelity", estimated_op, target_op)
        gate_metrics[str(label)] = {
            "name": _pretty_gate_name(label),
            "average_gate_infidelity": _call_metric("average_gate_infidelity", estimated_op, target_op),
            "entanglement_infidelity": infidelity,
            "entanglement_fidelity": None if infidelity is None else 1.0 - infidelity,
            "diamond_distance": _call_metric("diamond_distance", estimated_op, target_op),
        }
    return gate_metrics


def run_gst_analysis_2q(
    ds: xr.Dataset,
    design: GST2QExperimentDesign,
    pair_name: str,
    log_callable: Callable = print,
) -> tuple[object, dict[str, Any]]:
    """Run pyGSTi StandardGST on two-qubit counts and extract gate metrics."""
    gst_ds = transform_dataset_to_gst_2q(ds, design)
    gst_data = pygsti.protocols.ProtocolData(design.exp_design, gst_ds)
    gst_protocol = pygsti.protocols.StandardGST()
    gst_results = gst_protocol.run(gst_data, disable_checkpointing=True)

    estimates = getattr(gst_results, "estimates", {})
    if not estimates:
        raise RuntimeError("StandardGST returned no estimates.")

    preferred = ["full TP", "CPTPLND", "Target"]
    summary: dict[str, Any] = {}
    np.set_printoptions(suppress=True, precision=6)

    for key in preferred:
        if key not in estimates:
            continue
        est_model = _estimate_model(estimates[key])
        metrics = extract_gate_metrics(est_model, design.std_model)
        summary[key] = metrics
        log_callable(f"\n=== Under {key} ({pair_name}) ===")
        for gate_label, vals in metrics.items():
            inf = vals.get("entanglement_infidelity")
            fid = vals.get("entanglement_fidelity")
            if inf is None:
                log_callable(f"  {vals['name']} ({gate_label}): metrics unavailable")
            else:
                log_callable(f"  {vals['name']} infidelity: {inf:.6f}  fidelity: {fid:.6f}")

        try:
            rho_est = _pp_vector_to_stdmx(est_model.preps["rho0"])
            rho_std = _pp_vector_to_stdmx(design.std_model.preps["rho0"])
            state_fidelity = pygsti.tools.fidelity(rho_est, rho_std)
            summary.setdefault("spam", {})[key] = {"rho0_fidelity": float(state_fidelity)}
            log_callable(f"  State preparation fidelity: {state_fidelity:.6f}")
        except Exception as exc:
            log_callable(f"  Could not extract SPAM fidelity: {exc}")

    if "full TP" not in estimates:
        first_key = next(iter(estimates))
        est_model = _estimate_model(estimates[first_key])
        summary[first_key] = extract_gate_metrics(est_model, design.std_model)

    return gst_results, summary


def write_gst_html_report(
    gst_results,
    pair_name: str,
    snapshot_idx: int,
    report_dirname: Optional[str] = None,
) -> tuple[str, str]:
    """Write the pyGSTi HTML report next to the saved node data."""
    report_dirname = report_dirname or f"gst_report_{pair_name}"
    qs = get_qualibrate_config(get_qualibrate_config_path())
    node_dir = Path(get_node_dir_path(snapshot_idx, qs.storage.location))
    gst_report_dir = node_dir / report_dirname

    try:
        report = pygsti.report.construct_standard_report(
            gst_results,
            title=f"GST Report - {pair_name} (2Q CZ)",
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


def analyse_gst_data_2q(
    node: QualibrationNode,
    ds: xr.Dataset,
    design: GST2QExperimentDesign,
) -> Dict[str, object]:
    """Run 2Q GST analysis for the selected pair and store results on the node."""
    qubit_pairs = node.namespace["qubit_pairs"]
    node.results["gst_results"] = {}
    gst_result_objects = {}

    for qp in qubit_pairs:
        gst_results, gst_results_dict = run_gst_analysis_2q(
            ds,
            design,
            qp.name,
            log_callable=node.log,
        )
        node.results["gst_results"][qp.name] = gst_results_dict
        gst_result_objects[qp.name] = gst_results

    return gst_result_objects
