"""Full-flow integration test for virtual gate calibration (nodes 00-03).

Exercises the complete calibration pipeline on a 2-dot + 1-sensor + 1-barrier
device, building a compensation matrix:

    sensor  dot_1  dot_2  barrier
    [1      a_s1   a_s2   a_sB   ]   <- node 01 (sensor compensation)
    [0      T_11   T_12   g_1B   ]   <- node 02 (virtual plunger)
    [0      T_21   T_22   g_2B   ]   <- node 02 (virtual plunger)
    [b_Bs   b_B1   b_B2   1.0    ]   <- node 03 (barrier compensation)

TODO: re-enable sensor-vs-plunger scans in step 02 (a_1s, a_2s) once
edge detection handles the sensor column geometry reliably.

Uses qarray (with an 8th barrier gate) for charge stability simulations
(nodes 00, 01, 02) and HybridBarrierVirtualizationSimulator for tunnel
coupling data (node 03).  A final virtual gate sweep (step 04) compares
physical vs fully-compensated dot_1-vs-dot_2 scans.
"""

from __future__ import annotations

import base64
import io
import os
import sys
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
from unittest.mock import patch

# ── Compatibility shims ────────────────────────────────────────────────
if "qualibrate.parameters" not in sys.modules:
    try:
        import qualibrate.core.parameters as _cp
        sys.modules["qualibrate.parameters"] = _cp
    except ImportError:
        pass

os.environ.setdefault("QUAM_STATE_PATH", "/tmp/quam_test_state")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

# ── Path setup ─────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[5]
_CALIBRATION_LIBRARY_ROOT = (
    _REPO_ROOT / "qualibration_graphs" / "quantum_dots" / "calibrations" / "gate_virtualization"
)
_GATE_VIRT_UTILS = (
    _REPO_ROOT / "qualibration_graphs" / "quantum_dots" / "calibration_utils" / "gate_virtualization"
)
_QD_ROOT = _REPO_ROOT / "qualibration_graphs" / "quantum_dots"
_TEST_QD_ROOT = _REPO_ROOT / "tests" / "analysis" / "qualibration_graphs" / "quantum_dots"
_ARTIFACTS_BASE = _REPO_ROOT / "tests" / "analysis" / "artifacts" / "full_flow_virtual_gate"

_SHARED_DIR = _REPO_ROOT / "qualibration_graphs" / "quantum_dots" / "calibrations"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))
if str(_QD_ROOT) not in sys.path:
    sys.path.insert(0, str(_QD_ROOT))


# ── Module loading ─────────────────────────────────────────────────────

def _load_module(name: str, filepath: Path):
    spec = spec_from_file_location(name, filepath)
    mod = module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_sensor_analysis = _load_module("_full_flow_sd_analysis", _GATE_VIRT_UTILS / "sensor_dot_analysis.py")
_analysis_mod = _load_module("_full_flow_gv_analysis", _GATE_VIRT_UTILS / "analysis.py")
_barrier_analysis_mod = _load_module(
    "_full_flow_barrier_analysis", _GATE_VIRT_UTILS / "barrier_compensation_analysis.py"
)
_simulator_mod = _load_module(
    "_full_flow_hybrid_sim",
    _TEST_QD_ROOT / "calibration_utils" / "gate_virtualization" / "hybrid_barrier_virtualization_simulator.py",
)

fit_lorentzian = _sensor_analysis.fit_lorentzian
optimal_operating_point = _sensor_analysis.optimal_operating_point
process_raw_dataset = _analysis_mod.process_raw_dataset
HybridBarrierSimulationConfig = _simulator_mod.HybridBarrierSimulationConfig
HybridBarrierVirtualizationSimulator = _simulator_mod.HybridBarrierVirtualizationSimulator
finite_temperature_excess_charge = _barrier_analysis_mod.finite_temperature_excess_charge

from shared_fixtures import (
    apply_param_overrides,
    call_node_action,
    ensure_qua_dashboards_stub,
    patch_action_manager_register_only,
    patch_qualibrate_logger,
    reimport_node_to_register_actions,
    setup_test_cache,
)
from quam_factory import create_qd_quam
from validation_utils.charge_stability.default import InitDotModel, init_dot_model

from .conftest import (
    simulate_plunger_plunger_scan,
    simulate_sensor_sweep,
    sweep_voltages_mV,
)
from .simulation_helpers import simulate_sensor_device_scan

# ── Cache setup ────────────────────────────────────────────────────────

_CACHE_BASE = _REPO_ROOT / "tests" / "analysis" / ".pytest_cache"
setup_test_cache(_CACHE_BASE)
patch_qualibrate_logger(_CACHE_BASE)


# ── Gate names (from create_qd_quam factory) ───────────────────────────

SENSOR_GATE = "virtual_sensor_1"
DOT_1_GATE = "virtual_dot_1"
DOT_2_GATE = "virtual_dot_2"
BARRIER_GATE = "virtual_barrier_1"

PAIR_NAME = "virtual_dot_1_virtual_dot_2_pair"

GATE_ORDER = [SENSOR_GATE, DOT_1_GATE, DOT_2_GATE, BARRIER_GATE]

# qarray model: 6 dots -> gates 0-5, 1 sensor -> gate 6, 1 barrier -> gate 7.
GATE_TO_QARRAY_IDX = {
    DOT_1_GATE: 0,
    DOT_2_GATE: 1,
    SENSOR_GATE: 6,
    BARRIER_GATE: 7,
}

# Scan parameters
SENSOR_CENTER_MV = 5.0
SENSOR_SPAN_MV = 6.0
SENSOR_POINTS = 300

SENSOR_COMP_SPAN_V = 0.010
SENSOR_COMP_POINTS = 100
DEVICE_COMP_SPAN_V = 0.050
DEVICE_COMP_POINTS = 60

PLUNGER_CENTER_V = 0.0
PLUNGER_SPAN_V = 0.050
PLUNGER_POINTS = 200

BARRIER_PLUNGER_SPAN_V = 0.100


# ── DCChannelTracker ───────────────────────────────────────────────────


class DCChannelTracker:
    """Tracks DC operating points and sensor compensation across calibration steps."""

    def __init__(self, gate_map: dict[str, int]):
        self._gate_idx = dict(gate_map)
        self._dc_mV: dict[str, float] = {name: 0.0 for name in gate_map}
        self._sensor_comp: dict[str, dict[str, float]] = {}

    def set_dc(self, gate: str, voltage_mV: float) -> None:
        self._dc_mV[gate] = float(voltage_mV)

    def get_dc(self, gate: str) -> float:
        return self._dc_mV[gate]

    def set_sensor_compensation(self, sensor: str, gate: str, alpha: float) -> None:
        self._sensor_comp.setdefault(sensor, {})[gate] = float(alpha)

    def get_base_voltages(self) -> np.ndarray:
        n = max(self._gate_idx.values()) + 1
        v = np.zeros(n)
        for name, idx in self._gate_idx.items():
            v[idx] = self._dc_mV.get(name, 0.0)
        return v

    def sensor_comp_by_idx(self, sensor: str) -> dict[int, float]:
        result = {}
        for gate_name, alpha in self._sensor_comp.get(sensor, {}).items():
            if gate_name in self._gate_idx:
                result[self._gate_idx[gate_name]] = alpha
        return result

    def gate_idx(self, gate_name: str) -> int:
        return self._gate_idx[gate_name]


# ── Helpers ────────────────────────────────────────────────────────────


def _qarray_available() -> bool:
    try:
        from qarray import DotArray
        m = DotArray(Cdd=[[0.1]], Cgd=[[0.1]], algorithm="default", implementation="jax")
        m.ground_state_open(np.array([[0.0], [0.1]]))
        return True
    except Exception:
        return False


def _load_node(node_name: str, machine) -> Any:
    from quam_config import Quam

    ensure_qua_dashboards_stub()
    with (
        patch.object(Quam, "load", return_value=machine),
        patch_action_manager_register_only(),
    ):
        node = reimport_node_to_register_actions(node_name, _CALIBRATION_LIBRARY_ROOT)
    if node is None:
        pytest.fail(f"Could not load node '{node_name}' from {_CALIBRATION_LIBRARY_ROOT}")
    node.machine = machine
    return node


def _run_node_actions(node, ds_raw_all: dict, param_overrides: dict | None = None) -> None:
    overrides = dict(param_overrides) if param_overrides else {}
    overrides["simulate"] = False
    apply_param_overrides(node, overrides)
    node.results["ds_raw_all"] = ds_raw_all
    call_node_action(node, "analyse_data")
    call_node_action(node, "plot_data")
    action_names = set(getattr(getattr(node, "_action_manager", None), "actions", {}).keys())
    if "update_state" in action_names:
        call_node_action(node, "update_state")
    if "update_virtual_gate_matrix" in action_names:
        call_node_action(node, "update_virtual_gate_matrix")
    call_node_action(node, "save_results")


def _get_matrix(machine) -> np.ndarray:
    layer = machine.virtual_gate_sets["main_qpu"].layers[0]
    return np.asarray(layer.matrix, dtype=float)


def _get_gate_index(machine, gate_name: str) -> int:
    layer = machine.virtual_gate_sets["main_qpu"].layers[0]
    return list(layer.source_gates).index(gate_name)


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _collect_node_figures(node) -> list[tuple[str, str]]:
    """Extract (label, base64_png) pairs from a node's result figures."""
    figures = []
    for label, fig in node.results.get("figures", {}).items():
        try:
            figures.append((label, _fig_to_base64(fig)))
        except Exception:
            pass
    return figures


def _matrix_snapshot_html(matrix: np.ndarray, gate_labels: list[str], title: str) -> str:
    """Render a compensation matrix as an HTML table."""
    rows = [f"<h4>{title}</h4>", '<table class="matrix">']
    rows.append("<tr><th></th>" + "".join(f"<th>{g}</th>" for g in gate_labels) + "</tr>")
    for i, row_label in enumerate(gate_labels):
        cells = []
        for j in range(len(gate_labels)):
            val = matrix[i, j]
            css = ' class="diag"' if i == j else (' class="zero"' if val == 0.0 else "")
            cells.append(f"<td{css}>{val:.6f}</td>")
        rows.append(f"<tr><th>{row_label}</th>{''.join(cells)}</tr>")
    rows.append("</table>")
    return "\n".join(rows)


def _build_html_report(
    sections: list[dict],
    final_matrix_b64: str,
    output_path: Path,
) -> None:
    """Write a self-contained HTML report with embedded plots."""
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head><meta charset='utf-8'>",
        "<title>Virtual Gate Calibration Report</title>",
        "<style>",
        "body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 1100px; "
        "margin: 0 auto; padding: 24px; background: #fafafa; color: #222; }",
        "h1 { border-bottom: 2px solid #2563eb; padding-bottom: 8px; }",
        "h2 { color: #2563eb; margin-top: 2em; }",
        "h3 { color: #1e40af; }",
        "h4 { margin: 8px 0 4px; }",
        ".step { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; "
        "padding: 20px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }",
        ".plots { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }",
        ".plots img { max-width: 480px; border: 1px solid #d1d5db; border-radius: 4px; }",
        "table.matrix { border-collapse: collapse; margin: 8px 0; font-size: 13px; }",
        "table.matrix th, table.matrix td { border: 1px solid #d1d5db; padding: 4px 8px; text-align: right; }",
        "table.matrix th { background: #f3f4f6; }",
        "table.matrix td.diag { background: #dbeafe; font-weight: bold; }",
        "table.matrix td.zero { color: #dc2626; }",
        ".param-table { font-size: 13px; margin: 8px 0; }",
        ".param-table td { padding: 2px 10px 2px 0; }",
        ".param-table td:first-child { font-weight: 600; color: #374151; }",
        ".summary { background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; "
        "padding: 16px; margin-top: 24px; }",
        "footer { margin-top: 32px; font-size: 12px; color: #9ca3af; text-align: center; }",
        "</style></head><body>",
        "<h1>Virtual Gate Calibration -- Full Flow Report</h1>",
        f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "<p>Device: 2 quantum dots + 1 sensor dot + 1 barrier gate</p>",
        '<p>Gates: <code>' + ', '.join(GATE_ORDER) + '</code></p>',
    ]

    for section in sections:
        html_parts.append('<div class="step">')
        html_parts.append(f"<h2>{section['title']}</h2>")
        html_parts.append(f"<p>{section['description']}</p>")

        if section.get("params"):
            html_parts.append('<table class="param-table">')
            for k, v in section["params"].items():
                html_parts.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
            html_parts.append("</table>")

        if section.get("results_text"):
            html_parts.append(f"<p><b>Results:</b> {section['results_text']}</p>")

        if section.get("matrix_html"):
            html_parts.append(section["matrix_html"])

        if section.get("figures"):
            html_parts.append('<div class="plots">')
            for label, b64 in section["figures"]:
                html_parts.append(
                    f'<div><img src="data:image/png;base64,{b64}" '
                    f'alt="{label}"><br><small>{label}</small></div>'
                )
            html_parts.append("</div>")

        html_parts.append("</div>")

    html_parts.append('<div class="summary">')
    html_parts.append("<h2>Final Compensation Matrix</h2>")
    html_parts.append(
        f'<img src="data:image/png;base64,{final_matrix_b64}" '
        f'alt="Final matrix" style="max-width:600px;">'
    )
    html_parts.append("</div>")

    html_parts.append(
        f"<footer>qualibration-graphs virtual gate calibration test &middot; "
        f"{datetime.now().strftime('%Y-%m-%d')}</footer>"
    )
    html_parts.append("</body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts), encoding="utf-8")


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.analysis
@pytest.mark.skipif(not _qarray_available(), reason="qarray/JAX not functional")
class TestFullFlowVirtualGateCalibration:
    """Full-flow integration test: nodes 00 -> 01 -> 02 -> 03."""

    @pytest.fixture(autouse=True)
    def setup(self):
        Cgd_ext = [row + [0.02 if i < 2 else 0.0] for i, row in enumerate(InitDotModel.dot_gate_capacitance())]
        Cgs_ext = [row + [0.003] for row in InitDotModel.gate_sensor_capacitance()]
        self.dot_model = init_dot_model(Cgd=Cgd_ext, Cgs=Cgs_ext)
        self.machine = create_qd_quam()
        self.dc = DCChannelTracker(GATE_TO_QARRAY_IDX)
        self.artifacts_dir = _ARTIFACTS_BASE
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.report_sections: list[dict] = []

    # ── Step 0: Sensor dot tuning ──────────────────────────────────

    def _step_00_sensor_tuning(self) -> float:
        v_sensor_mV = np.linspace(
            SENSOR_CENTER_MV - SENSOR_SPAN_MV / 2,
            SENSOR_CENTER_MV + SENSOR_SPAN_MV / 2,
            SENSOR_POINTS,
        )
        ds = simulate_sensor_sweep(
            self.dot_model, v_sensor_mV, base_voltages=self.dc.get_base_voltages(),
        )
        signal = np.hypot(ds["I"].values[0], ds["Q"].values[0])
        v_V = ds.coords["x_volts"].values

        result = fit_lorentzian(v_V, signal, side="right")
        optimal_mV = result.optimal_voltage * 1e3
        self.dc.set_dc(SENSOR_GATE, optimal_mV)

        matrix_before = _get_matrix(self.machine)
        np.testing.assert_allclose(
            matrix_before, np.eye(matrix_before.shape[0]), atol=1e-12,
            err_msg="Matrix should be identity before any calibration",
        )

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(v_V * 1e3, signal, "k-", lw=0.8)
        ax.axvline(optimal_mV, color="red", ls="--", lw=1, label=f"optimal = {optimal_mV:.2f} mV")
        ax.set_xlabel("Sensor gate (mV)")
        ax.set_ylabel("Signal (a.u.)")
        ax.set_title("Node 00: Sensor Dot Tuning")
        ax.legend(fontsize=8)
        fig.tight_layout()

        self.report_sections.append({
            "title": "Step 0: Sensor Dot Tuning (Node 00)",
            "description": (
                "Sweep the sensor plunger gate across a Coulomb peak, fit a Lorentzian, "
                "and set the DC operating point to the steepest slope (maximum sensitivity)."
            ),
            "params": {
                "Sweep center": f"{SENSOR_CENTER_MV:.1f} mV",
                "Sweep span": f"{SENSOR_SPAN_MV:.1f} mV",
                "Points": str(SENSOR_POINTS),
            },
            "results_text": f"Optimal sensor operating point: <b>{optimal_mV:.3f} mV</b>",
            "figures": [("Sensor sweep + Lorentzian fit", _fig_to_base64(fig))],
        })
        plt.close(fig)

        return optimal_mV

    # ── Step 1: Sensor compensation ────────────────────────────────

    def _step_01_sensor_compensation(self) -> dict[str, float]:
        v_sensor = sweep_voltages_mV(0.0, SENSOR_COMP_SPAN_V, SENSOR_COMP_POINTS)
        v_device = sweep_voltages_mV(0.0, DEVICE_COMP_SPAN_V, DEVICE_COMP_POINTS)

        ds_raw_all = {}

        for gate_name in [DOT_1_GATE, DOT_2_GATE, BARRIER_GATE]:
            ds_raw = simulate_sensor_device_scan(
                self.dot_model, v_sensor, v_device,
                sensor_gate_idx=GATE_TO_QARRAY_IDX[SENSOR_GATE],
                device_gate_idx=GATE_TO_QARRAY_IDX[gate_name],
                sensor_operating_point=self.dc.get_dc(SENSOR_GATE),
                base_voltages=self.dc.get_base_voltages(),
            )
            ds_raw_all[f"{SENSOR_GATE}_vs_{gate_name}"] = ds_raw

        node = _load_node("01_sensor_gate_compensation", self.machine)
        _run_node_actions(node, ds_raw_all, param_overrides={
            "sensor_gate_span": SENSOR_COMP_SPAN_V,
            "sensor_gate_points": SENSOR_COMP_POINTS,
            "device_gate_span": DEVICE_COMP_SPAN_V,
            "device_gate_points": DEVICE_COMP_POINTS,
            "sensor_device_mapping": {SENSOR_GATE: [DOT_1_GATE, DOT_2_GATE, BARRIER_GATE]},
        })

        alphas = {}
        for pair_key, fit in node.results["fit_results"].items():
            _, device_gate = pair_key.split("_vs_")
            alpha = fit["coefficient"]
            assert np.isfinite(alpha), f"Alpha for {pair_key} is not finite"
            alphas[device_gate] = alpha
            self.dc.set_sensor_compensation(SENSOR_GATE, device_gate, alpha)

        sensor_idx = _get_gate_index(self.machine, SENSOR_GATE)
        matrix = _get_matrix(self.machine)
        for gate_name, alpha in alphas.items():
            col_idx = _get_gate_index(self.machine, gate_name)
            assert matrix[sensor_idx, col_idx] == pytest.approx(alpha), (
                f"Matrix[{SENSOR_GATE}, {gate_name}] mismatch"
            )

        gate_labels = [
            [k for k, v in {g: _get_gate_index(self.machine, g) for g in GATE_ORDER}.items() if v == idx][0]
            for idx in sorted(_get_gate_index(self.machine, g) for g in GATE_ORDER)
        ]
        focus_idx = sorted(_get_gate_index(self.machine, g) for g in GATE_ORDER)
        self.report_sections.append({
            "title": "Step 1: Sensor Gate Compensation (Node 01)",
            "description": (
                "For each device gate (dot_1, dot_2, barrier), sweep sensor vs device gate "
                "in a 2D scan, fit a shifted Lorentzian to extract the cross-talk coefficient "
                "alpha = dV_sensor / dV_device. These populate the sensor row of the "
                "compensation matrix."
            ),
            "params": {
                "Sensor span": f"{SENSOR_COMP_SPAN_V * 1e3:.1f} mV ({SENSOR_COMP_POINTS} pts)",
                "Device span": f"{DEVICE_COMP_SPAN_V * 1e3:.1f} mV ({DEVICE_COMP_POINTS} pts)",
                "Device gates": ", ".join([DOT_1_GATE, DOT_2_GATE, BARRIER_GATE]),
            },
            "results_text": ", ".join(f"alpha({g}) = {a:.6f}" for g, a in alphas.items()),
            "matrix_html": _matrix_snapshot_html(
                matrix[np.ix_(focus_idx, focus_idx)], gate_labels,
                "Compensation matrix after Node 01",
            ),
            "figures": _collect_node_figures(node),
        })

        return alphas

    # ── Step 2: Virtual plunger calibration (5 runs) ───────────────

    def _step_02_virtual_plunger(self) -> dict[str, Any]:
        sensor_comp = self.dc.sensor_comp_by_idx(SENSOR_GATE)
        sensor_op_mV = self.dc.get_dc(SENSOR_GATE)
        base_voltages = self.dc.get_base_voltages()
        results = {}
        all_figures: list[tuple[str, str]] = []

        scan_configs = [
            # (plunger_gate, device_gate, x_span, x_pts, y_span, y_pts)
            (DOT_1_GATE, DOT_2_GATE, PLUNGER_SPAN_V, PLUNGER_POINTS, PLUNGER_SPAN_V, PLUNGER_POINTS),
            (DOT_1_GATE, BARRIER_GATE, PLUNGER_SPAN_V, PLUNGER_POINTS, BARRIER_PLUNGER_SPAN_V, PLUNGER_POINTS),
            (DOT_2_GATE, BARRIER_GATE, PLUNGER_SPAN_V, PLUNGER_POINTS, BARRIER_PLUNGER_SPAN_V, PLUNGER_POINTS),
            # TODO: re-enable sensor-vs-plunger scans once edge detection handles
            #       the sensor column geometry reliably.
            # (DOT_1_GATE, SENSOR_GATE, PLUNGER_SPAN_V, PLUNGER_POINTS, PLUNGER_SPAN_V, PLUNGER_POINTS),
            # (DOT_2_GATE, SENSOR_GATE, PLUNGER_SPAN_V, PLUNGER_POINTS, PLUNGER_SPAN_V, PLUNGER_POINTS),
        ]

        for plunger_gate, device_gate, x_span, x_pts, y_span, y_pts in scan_configs:
            v_px = sweep_voltages_mV(PLUNGER_CENTER_V, x_span, x_pts)
            v_py = sweep_voltages_mV(PLUNGER_CENTER_V, y_span, y_pts)

            ds_raw = simulate_plunger_plunger_scan(
                self.dot_model, v_px, v_py,
                plunger_x_gate_idx=GATE_TO_QARRAY_IDX[plunger_gate],
                plunger_y_gate_idx=GATE_TO_QARRAY_IDX[device_gate],
                sensor_operating_point=sensor_op_mV,
                base_voltages=base_voltages,
                sensor_compensation=sensor_comp,
            )

            pair_key = f"{plunger_gate}_vs_{device_gate}"
            node = _load_node("02_virtual_plunger_calibration", self.machine)
            _run_node_actions(node, {pair_key: ds_raw}, param_overrides={
                "plunger_gate_span": x_span,
                "plunger_gate_points": x_pts,
                "device_gate_span": y_span,
                "device_gate_points": y_pts,
                "plunger_device_mapping": {plunger_gate: [device_gate]},
            })

            fit = node.results["fit_results"][pair_key]
            assert fit["fit_params"]["success"], f"Fit failed for {pair_key}"
            T = fit["T_matrix"]
            assert T is not None
            assert abs(np.linalg.det(T)) > 1e-6, f"T singular for {pair_key}"
            results[pair_key] = fit
            all_figures.extend(
                (f"{pair_key}: {label}", b64)
                for label, b64 in _collect_node_figures(node)
            )

        matrix = _get_matrix(self.machine)
        gate_labels = [
            [k for k, v in {g: _get_gate_index(self.machine, g) for g in GATE_ORDER}.items() if v == idx][0]
            for idx in sorted(_get_gate_index(self.machine, g) for g in GATE_ORDER)
        ]
        focus_idx = sorted(_get_gate_index(self.machine, g) for g in GATE_ORDER)
        scan_names = [f"{pg} vs {dg}" for pg, dg, *_ in scan_configs]
        self.report_sections.append({
            "title": "Step 2: Virtual Plunger Calibration (Node 02)",
            "description": (
                f"Run {len(scan_configs)} charge stability scans to extract T-matrix elements "
                "that relate each pair of virtual gates. Scans: " + "; ".join(scan_names) + ". "
                "Each scan uses BayesianCP edge detection to find charge transition lines "
                "and extract the slope ratio (T-matrix). Results populate the plunger-plunger "
                "block and plunger-barrier coupling columns."
            ),
            "params": {
                "Plunger span": f"{PLUNGER_SPAN_V * 1e3:.1f} mV ({PLUNGER_POINTS} pts)",
                "Barrier span": f"{BARRIER_PLUNGER_SPAN_V * 1e3:.1f} mV ({PLUNGER_POINTS} pts)",
                "Scans": str(len(scan_configs)),
            },
            "results_text": ", ".join(
                f"det(T[{k}]) = {abs(np.linalg.det(v['T_matrix'])):.4f}" for k, v in results.items()
            ),
            "matrix_html": _matrix_snapshot_html(
                matrix[np.ix_(focus_idx, focus_idx)], gate_labels,
                "Compensation matrix after Node 02",
            ),
            "figures": all_figures,
        })

        return results

    # ── Step 3: Barrier compensation with non-barrier drives ───────

    def _step_03_barrier_compensation(self) -> dict[str, float]:
        barrier_names = [BARRIER_GATE]
        base_tunnel = np.array([0.080], dtype=float)
        self_gamma = 0.75 / base_tunnel[0]

        dot1_gamma = -0.15 / base_tunnel[0]
        dot2_gamma = -0.18 / base_tunnel[0]
        sensor_gamma = -0.05 / base_tunnel[0]

        drive_values = np.linspace(-0.022, 0.022, 7, dtype=float)
        detuning_values = np.linspace(-1.0, 1.0, 181, dtype=float)

        sim_cfg = HybridBarrierSimulationConfig(
            barrier_names=barrier_names,
            barrier_exponent_matrix=np.array([[self_gamma]]),
            base_tunnel_couplings=base_tunnel,
            drive_values=drive_values,
            detuning_values=detuning_values,
            thermal_energy=0.04,
            qarray_background_weight=0.0,
            analytic_background_weight=0.0,
            noise_std=0.00005,
            random_seed=42,
            use_qarray_background=False,
        )
        simulator = HybridBarrierVirtualizationSimulator(sim_cfg)

        pair_resolution = [
            {
                "target_pair_id": PAIR_NAME,
                "target_barrier": BARRIER_GATE,
                "drive_barriers": [BARRIER_GATE],
            }
        ]
        ds_raw_all, truth = simulator.generate_campaign_from_pair_resolution(pair_resolution)

        nb_drive_values = np.linspace(-0.040, 0.040, 7, dtype=float)
        nb_gammas = {DOT_1_GATE: dot1_gamma, DOT_2_GATE: dot2_gamma, SENSOR_GATE: sensor_gamma}

        for gate_name, gamma_val in nb_gammas.items():
            tunnel_vs_drive = base_tunnel[0] * np.exp(gamma_val * nb_drive_values)
            signal_2d = np.empty((len(nb_drive_values), len(detuning_values)), dtype=float)
            for row_idx, t_val in enumerate(tunnel_vs_drive):
                transition = finite_temperature_excess_charge(detuning_values, t_val, 0.04, 0.0)
                signal_2d[row_idx, :] = 0.25 + 1.10 * transition

            rng = np.random.default_rng(hash(gate_name) % (2**31))
            signal_2d += rng.normal(0, 0.00005, size=signal_2d.shape)

            I_data = signal_2d[np.newaxis, :, :]
            Q_data = np.zeros_like(I_data)
            ds = xr.Dataset(
                {
                    "I": xr.DataArray(
                        I_data,
                        dims=["sensors", "drive_volts", "detuning_volts"],
                        coords={
                            "sensors": ["sensor_1"],
                            "drive_volts": nb_drive_values,
                            "detuning_volts": detuning_values,
                        },
                    ),
                    "Q": xr.DataArray(
                        Q_data,
                        dims=["sensors", "drive_volts", "detuning_volts"],
                        coords={
                            "sensors": ["sensor_1"],
                            "drive_volts": nb_drive_values,
                            "detuning_volts": detuning_values,
                        },
                    ),
                }
            )
            pair_key = f"{PAIR_NAME}__{BARRIER_GATE}_vs_{gate_name}"
            ds_raw_all[pair_key] = ds

        # Build metadata for all datasets
        pair_metadata = {}
        for pair_key in ds_raw_all:
            parts = pair_key.split("__")
            if len(parts) == 2:
                target_pair_id = parts[0]
                rest = parts[1]
            else:
                target_pair_id = PAIR_NAME
                rest = pair_key
            target_barrier, drive_gate = rest.split("_vs_")
            is_nb = drive_gate in nb_gammas
            pair_metadata[pair_key] = {
                "target_pair_id": target_pair_id,
                "target_barrier": target_barrier,
                "drive_barrier": drive_gate,
                "drive_type": "non_barrier" if is_nb else "barrier",
                "drive_axis": "drive_volts",
                "detuning_axis": "detuning_volts",
                "detuning_axis_name": "detuning",
                "drive_center": 0.0,
                "detuning_center": 0.0,
                "dot_ids": [DOT_1_GATE, DOT_2_GATE],
                "drive_barriers": [BARRIER_GATE],
                "drive_non_barriers": list(nb_gammas.keys()) if is_nb else [],
            }

        node = _load_node("03_barrier_compensation", self.machine)
        overrides = {
            "simulate": False,
            "pair_names": [PAIR_NAME],
            "slope_sweep_span_mv": 44.0,
            "slope_sweep_points": 7,
            "min_slope_snr": 0.05,
            "min_tunnel_span_sigma": 0.01,
            "min_pair_fit_r2": 0.0,
            "include_non_barrier_drives": True,
            "non_barrier_drive_span_mv": 80.0,
            "non_barrier_drive_points": 7,
        }
        apply_param_overrides(node, overrides)

        node.results["ds_raw_all"] = ds_raw_all
        node.namespace["pair_metadata"] = pair_metadata
        node.namespace["barrier_order"] = [BARRIER_GATE]
        node.namespace["pair_resolution"] = [pair_metadata[next(iter(pair_metadata))]]
        node.namespace["drive_mapping"] = {BARRIER_GATE: [BARRIER_GATE]}
        node.namespace["target_barrier_to_pair_id"] = {BARRIER_GATE: PAIR_NAME}

        call_node_action(node, "analyse_data")
        call_node_action(node, "plot_data")
        call_node_action(node, "update_virtual_gate_matrix")
        call_node_action(node, "save_results")

        barrier_idx = _get_gate_index(self.machine, BARRIER_GATE)
        matrix = _get_matrix(self.machine)
        assert np.isfinite(matrix[barrier_idx, barrier_idx]), "Barrier diagonal is not finite"
        assert matrix[barrier_idx, barrier_idx] != 0.0, "Barrier diagonal is zero"

        betas = node.results.get("non_barrier_betas", {}).get(BARRIER_GATE, {})
        for gate_name in [DOT_1_GATE, DOT_2_GATE, SENSOR_GATE]:
            col_idx = _get_gate_index(self.machine, gate_name)
            val = matrix[barrier_idx, col_idx]
            assert np.isfinite(val), f"Matrix[{BARRIER_GATE}, {gate_name}] is not finite"
            assert val != 0.0, f"Matrix[{BARRIER_GATE}, {gate_name}] is zero"

        gate_labels = [
            [k for k, v in {g: _get_gate_index(self.machine, g) for g in GATE_ORDER}.items() if v == idx][0]
            for idx in sorted(_get_gate_index(self.machine, g) for g in GATE_ORDER)
        ]
        focus_idx = sorted(_get_gate_index(self.machine, g) for g in GATE_ORDER)
        beta_strs = [f"beta({g}) = {b:.6f}" for g, b in betas.items()]
        self.report_sections.append({
            "title": "Step 3: Barrier Compensation (Node 03)",
            "description": (
                "Measure tunnel coupling vs gate voltage for the barrier self-drive and "
                "for non-barrier drives (plungers, sensor). Fit a finite-temperature "
                "two-level model to extract dt/dV slopes. Barrier self-coupling normalizes "
                "to 1.0 on the diagonal; non-barrier beta = (dt/dV_gate) / (dt/dV_barrier) "
                "populates the barrier row off-diagonals."
            ),
            "params": {
                "Barrier sweep": f"{44.0:.0f} mV, {7} pts",
                "Non-barrier sweep": f"{80.0:.0f} mV, {7} pts",
                "Non-barrier drives": ", ".join(nb_gammas.keys()),
            },
            "results_text": "; ".join(beta_strs) if beta_strs else "No non-barrier betas",
            "matrix_html": _matrix_snapshot_html(
                matrix[np.ix_(focus_idx, focus_idx)], gate_labels,
                "Compensation matrix after Node 03 (final)",
            ),
            "figures": _collect_node_figures(node),
        })

        return dict(betas)

    # ── Step 4: Virtual gate sweep (physical vs virtual comparison) ─

    def _step_04_virtual_gate_sweep(self) -> dict[str, np.ndarray]:
        """Sweep virtual_dot_1 vs virtual_dot_2 through the full compensation
        matrix and compare against a raw physical sweep to visualise the
        effect of plunger-plunger and plunger-sensor compensation."""
        matrix = _get_matrix(self.machine)
        layer = self.machine.virtual_gate_sets["main_qpu"].layers[0]
        source_gates = list(layer.source_gates)
        n_gates_matrix = len(source_gates)

        matrix_idx = {g: source_gates.index(g) for g in GATE_ORDER}
        qarray_idx = dict(GATE_TO_QARRAY_IDX)

        midx_to_qidx: dict[int, int] = {}
        for g in GATE_ORDER:
            midx_to_qidx[matrix_idx[g]] = qarray_idx[g]

        dot1_mi = matrix_idx[DOT_1_GATE]
        dot2_mi = matrix_idx[DOT_2_GATE]

        v_sweep = sweep_voltages_mV(0.0, PLUNGER_SPAN_V, PLUNGER_POINTS)
        base_v = self.dc.get_base_voltages()

        def _simulate_sweep(comp_matrix: np.ndarray) -> np.ndarray:
            rows = []
            for vy in v_sweep:
                for vx in v_sweep:
                    virt = np.zeros(n_gates_matrix)
                    virt[dot1_mi] = vx
                    virt[dot2_mi] = vy
                    phys = comp_matrix @ virt
                    v = base_v.copy()
                    for mi, qi in midx_to_qidx.items():
                        v[qi] += phys[mi]
                    rows.append(v)
            va = np.array(rows)
            z, _ = self.dot_model.charge_sensor_open(-va)
            return z.squeeze().reshape(len(v_sweep), len(v_sweep))

        identity = np.eye(n_gates_matrix)

        sensor_only = identity.copy()
        sensor_mi = matrix_idx[SENSOR_GATE]
        sensor_only[sensor_mi, :] = matrix[sensor_mi, :]

        data_physical = _simulate_sweep(identity)
        data_sensor_comp = _simulate_sweep(sensor_only)
        data_virtual = _simulate_sweep(matrix)

        extent = [v_sweep[0], v_sweep[-1], v_sweep[0], v_sweep[-1]]
        vmin = min(data_physical.min(), data_virtual.min())
        vmax = max(data_physical.max(), data_virtual.max())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = [
            "Physical (no compensation)",
            "Sensor-only compensation",
            "Full virtual gate compensation",
        ]
        datasets = [data_physical, data_sensor_comp, data_virtual]

        for ax, title, data in zip(axes, titles, datasets):
            im = ax.imshow(
                data, extent=extent, origin="lower", aspect="auto",
                cmap="hot", vmin=vmin, vmax=vmax,
            )
            ax.set_xlabel("dot_1 (mV)")
            ax.set_ylabel("dot_2 (mV)")
            ax.set_title(title, fontsize=10)
            bg_std = np.std(data[80:120, :])
            ax.text(
                0.02, 0.02, f"bg_std={bg_std:.4f}",
                transform=ax.transAxes, fontsize=8, color="white",
                va="bottom", bbox=dict(boxstyle="round", fc="black", alpha=0.6),
            )
        fig.colorbar(im, ax=axes, shrink=0.8, label="Sensor signal (a.u.)")
        fig.suptitle(
            "Virtual Gate Sweep: dot_1 vs dot_2", fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save_figure(fig, self.artifacts_dir / "virtual_gate_sweep_comparison.png")

        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        for ax, title, data in zip(axes2, titles, datasets):
            im2 = ax.imshow(
                data, extent=extent, origin="lower", aspect="auto", cmap="hot",
            )
            ax.set_xlabel("dot_1 (mV)")
            ax.set_ylabel("dot_2 (mV)")
            ax.set_title(title, fontsize=10)
            bg_std = np.std(data[80:120, :])
            ax.text(
                0.02, 0.02, f"bg_std={bg_std:.4f}",
                transform=ax.transAxes, fontsize=8, color="white",
                va="bottom", bbox=dict(boxstyle="round", fc="black", alpha=0.6),
            )
        fig2.colorbar(im2, ax=axes2, shrink=0.8, label="Sensor signal (a.u.)")
        fig2.suptitle(
            "Virtual Gate Sweep (auto-scaled): dot_1 vs dot_2",
            fontsize=12, fontweight="bold",
        )
        fig2.tight_layout(rect=[0, 0, 1, 0.95])

        report_figs = [
            ("Virtual gate sweep (shared scale)", _fig_to_base64(fig)),
            ("Virtual gate sweep (auto-scaled)", _fig_to_base64(fig2)),
        ]
        plt.close(fig2)

        self.report_sections.append({
            "title": "Step 4: Virtual Gate Sweep Comparison",
            "description": (
                "Sweep dot_1 vs dot_2 three ways: (1) physical gates with no "
                "compensation, (2) sensor-only compensation (step 01 alphas), "
                "(3) full virtual gate compensation (complete matrix from all "
                "calibration steps). The full compensation should orthogonalise "
                "the charge transition lines and flatten the sensor background."
            ),
            "params": {
                "Sweep span": f"{PLUNGER_SPAN_V * 1e3:.1f} mV ({PLUNGER_POINTS} pts)",
                "Sensor operating point": f"{self.dc.get_dc(SENSOR_GATE):.2f} mV",
            },
            "results_text": (
                f"bg_std: physical={np.std(data_physical[80:120, :]):.4f}, "
                f"sensor_comp={np.std(data_sensor_comp[80:120, :]):.4f}, "
                f"full_virtual={np.std(data_virtual[80:120, :]):.4f}"
            ),
            "figures": report_figs,
        })

        return {
            "physical": data_physical,
            "sensor_comp": data_sensor_comp,
            "virtual": data_virtual,
        }

    # ── Main test ──────────────────────────────────────────────────

    def test_full_flow(self):
        """Run the complete calibration flow and verify the final matrix."""
        # Step 0: Sensor tuning
        sensor_opt = self._step_00_sensor_tuning()
        assert np.isfinite(sensor_opt)

        # Step 1: Sensor compensation (dot_1, dot_2, barrier)
        alphas = self._step_01_sensor_compensation()
        assert len(alphas) == 3

        # Step 2: Virtual plunger (3 runs: dot-dot, dot-barrier x2)
        plunger_results = self._step_02_virtual_plunger()
        assert len(plunger_results) == 3

        # Step 3: Barrier compensation with non-barrier drives
        betas = self._step_03_barrier_compensation()

        # Step 4: Virtual gate sweep (physical vs virtual comparison)
        sweep_data = self._step_04_virtual_gate_sweep()
        assert "physical" in sweep_data and "virtual" in sweep_data

        # ── Final verification ─────────────────────────────────────
        matrix = _get_matrix(self.machine)

        gate_indices = {
            SENSOR_GATE: _get_gate_index(self.machine, SENSOR_GATE),
            DOT_1_GATE: _get_gate_index(self.machine, DOT_1_GATE),
            DOT_2_GATE: _get_gate_index(self.machine, DOT_2_GATE),
            BARRIER_GATE: _get_gate_index(self.machine, BARRIER_GATE),
        }
        focus_indices = sorted(gate_indices.values())
        focus_matrix = matrix[np.ix_(focus_indices, focus_indices)]

        # Sensor-vs-plunger scans are disabled (TODO), so the plunger rows
        # will have zeros in the sensor column.  Skip those cells.
        sensor_local = focus_indices.index(gate_indices[SENSOR_GATE])
        skip_cells = {
            (focus_indices.index(gate_indices[DOT_1_GATE]), sensor_local),
            (focus_indices.index(gate_indices[DOT_2_GATE]), sensor_local),
        }
        for i_local, i_global in enumerate(focus_indices):
            for j_local, j_global in enumerate(focus_indices):
                if (i_local, j_local) in skip_cells:
                    continue
                val = focus_matrix[i_local, j_local]
                gate_row = [k for k, v in gate_indices.items() if v == i_global][0]
                gate_col = [k for k, v in gate_indices.items() if v == j_global][0]
                assert val != 0.0, (
                    f"Matrix[{gate_row}, {gate_col}] is zero -- matrix not fully populated"
                )

        assert abs(np.linalg.det(focus_matrix)) > 1e-10, (
            f"Focus matrix is singular: det = {np.linalg.det(focus_matrix)}"
        )

        focus_inv = np.linalg.inv(focus_matrix)
        assert np.all(np.isfinite(focus_inv)), "C^{-1} has non-finite entries"

        # DC state preserved
        assert self.dc.get_dc(SENSOR_GATE) == pytest.approx(sensor_opt)
        for gate_name in [DOT_1_GATE, DOT_2_GATE, BARRIER_GATE]:
            assert gate_name in self.dc._sensor_comp.get(SENSOR_GATE, {})

        # Save matrix artifact
        gate_labels = [
            [k for k, v in gate_indices.items() if v == idx][0] for idx in focus_indices
        ]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(focus_matrix, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(len(gate_labels)))
        ax.set_xticklabels(gate_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(gate_labels)))
        ax.set_yticklabels(gate_labels, fontsize=8)
        for i_l in range(len(focus_indices)):
            for j_l in range(len(focus_indices)):
                ax.text(
                    j_l, i_l, f"{focus_matrix[i_l, j_l]:.4f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(focus_matrix[i_l, j_l]) > 0.5 else "black",
                )
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Final 4x4 Compensation Matrix")
        plt.tight_layout()
        final_matrix_b64 = _fig_to_base64(fig)
        _save_figure(fig, self.artifacts_dir / "final_matrix.png")
        assert (self.artifacts_dir / "final_matrix.png").exists()

        # Generate HTML report
        report_path = self.artifacts_dir / "calibration_report.html"
        _build_html_report(self.report_sections, final_matrix_b64, report_path)
        assert report_path.exists(), "HTML report was not generated"
