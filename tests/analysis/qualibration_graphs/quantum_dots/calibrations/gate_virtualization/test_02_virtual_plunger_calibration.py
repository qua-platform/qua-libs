"""Analysis tests for node 02 — virtual plunger calibration.

These tests simulate qarray datasets, inject ``ds_raw_all`` into the real node,
then execute the node run_actions (``analyse_data`` and ``plot_data``) through
``analysis_runner``.  The test module does not call node-specific analysis or
plotting utilities directly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from .conftest import (
    CALIBRATION_LIBRARY_ROOT,
    CALIBRATED_SENSOR_COMP,
    _call_node_action,
    _patch_action_manager_register_only,
    _reimport_node_to_register_actions,
    simulate_plunger_plunger_scan,
    simulate_sensor_sweep,
    sweep_voltages_mV,
)


# Gate names used by the test QuAM factory.
PLUNGER_X_GATE = "virtual_dot_1"
PLUNGER_Y_GATE = "virtual_dot_2"

PLUNGER_X_CENTER_V = 0.0
PLUNGER_X_SPAN_V = 0.050
PLUNGER_Y_CENTER_V = 0.0
PLUNGER_Y_SPAN_V = 0.050
PLUNGER_X_POINTS = 200
PLUNGER_Y_POINTS = 200

SENSOR_SWEEP_CENTER_MV = 5.0
SENSOR_SWEEP_SPAN_MV = 6.0
SENSOR_SWEEP_POINTS = 300

SENSOR_COMP = CALIBRATED_SENSOR_COMP


def _default_param_overrides(**extra):
    """Merge test-default node parameters with any caller overrides."""
    base = {
        "plunger_gate_span": PLUNGER_X_SPAN_V,
        "plunger_gate_points": PLUNGER_X_POINTS,
        "device_gate_span": PLUNGER_Y_SPAN_V,
        "device_gate_points": PLUNGER_Y_POINTS,
        "plunger_device_mapping": {PLUNGER_X_GATE: [PLUNGER_Y_GATE]},
    }
    base.update(extra)
    return base


def _qarray_available() -> bool:
    """Check whether qarray + JAX can actually run."""
    try:
        from qarray import DotArray

        model = DotArray(Cdd=[[0.1]], Cgd=[[0.1]], algorithm="default", implementation="jax")
        model.ground_state_open(np.array([[0.0], [0.1]]))
        return True
    except Exception:
        return False


def test_update_state_composes_all_rows(minimal_quam_factory):
    """update_state composes selected columns for all rows, including sensor row."""
    from quam_config import Quam

    machine = minimal_quam_factory()
    layer = machine.virtual_gate_sets["main_qpu"].layers[0]
    source_gates = list(layer.source_gates)
    sensor_gate = "virtual_sensor_1"

    sensor_idx = source_gates.index(sensor_gate)
    plunger_idx = source_gates.index(PLUNGER_X_GATE)
    device_idx = source_gates.index(PLUNGER_Y_GATE)
    cols = [plunger_idx, device_idx]

    # Seed a non-trivial existing matrix to verify composition (not replacement).
    old = np.asarray(layer.matrix, dtype=float)
    old[sensor_idx, plunger_idx] = -0.020
    old[sensor_idx, device_idx] = -0.035
    old[plunger_idx, plunger_idx] = 1.12
    old[plunger_idx, device_idx] = 0.08
    old[device_idx, plunger_idx] = -0.06
    old[device_idx, device_idx] = 0.93
    layer.matrix = old.tolist()

    delta = np.array(
        [
            [1.0, 0.14],
            [-0.09, 1.0],
        ],
        dtype=float,
    )
    expected = old.copy()
    expected[:, cols] = old[:, cols] @ delta

    with (
        patch.object(Quam, "load", return_value=machine),
        _patch_action_manager_register_only(),
    ):
        node = _reimport_node_to_register_actions(
            "02_virtual_plunger_calibration",
            CALIBRATION_LIBRARY_ROOT,
        )
    assert node is not None
    node.machine = machine
    node.results["fit_results"] = {
        f"{PLUNGER_X_GATE}_vs_{PLUNGER_Y_GATE}": {
            "fit_params": {"success": True},
            "T_matrix": delta,
        }
    }

    _call_node_action(node, "update_state")

    updated = np.asarray(
        machine.virtual_gate_sets["main_qpu"].layers[0].matrix,
        dtype=float,
    )
    np.testing.assert_allclose(updated[:, cols], expected[:, cols], atol=1e-12)

    other_cols = [j for j in range(updated.shape[1]) if j not in cols]
    if other_cols:
        np.testing.assert_allclose(updated[:, other_cols], old[:, other_cols], atol=1e-12)


@pytest.mark.analysis
@pytest.mark.skipif(not _qarray_available(), reason="qarray/JAX not functional in this environment")
class TestVirtualPlungerE2E:
    """End-to-end: qarray simulation -> node analyse_data/plot_data actions."""

    def _find_sensor_opt(self, dot_model) -> float:
        """Run a 1D sensor sweep and return the operating point in mV."""
        base_voltages = np.zeros(7)
        base_voltages[0] = PLUNGER_X_CENTER_V * 1e3
        base_voltages[1] = PLUNGER_Y_CENTER_V * 1e3

        v_sensor_mV = np.linspace(
            SENSOR_SWEEP_CENTER_MV - SENSOR_SWEEP_SPAN_MV / 2,
            SENSOR_SWEEP_CENTER_MV + SENSOR_SWEEP_SPAN_MV / 2,
            SENSOR_SWEEP_POINTS,
        )
        ds = simulate_sensor_sweep(dot_model, v_sensor_mV, base_voltages=base_voltages)
        signal = np.hypot(ds["I"].values[0], ds["Q"].values[0])
        v_sensor_V = ds.coords["x_volts"].values
        return float(v_sensor_V[int(np.argmax(signal))] * 1e3)

    def _run_node_pipeline(self, dot_model, analysis_runner):
        """Simulate one plunger-plunger scan and run node 02 analysis/plot actions."""
        sensor_opt_mV = self._find_sensor_opt(dot_model)

        v_px = sweep_voltages_mV(PLUNGER_X_CENTER_V, PLUNGER_X_SPAN_V, PLUNGER_X_POINTS)
        v_py = sweep_voltages_mV(PLUNGER_Y_CENTER_V, PLUNGER_Y_SPAN_V, PLUNGER_Y_POINTS)

        ds_raw = simulate_plunger_plunger_scan(
            dot_model,
            v_px,
            v_py,
            plunger_x_gate_idx=0,
            plunger_y_gate_idx=1,
            sensor_operating_point=sensor_opt_mV,
            sensor_compensation=SENSOR_COMP,
        )

        pair_key = f"{PLUNGER_X_GATE}_vs_{PLUNGER_Y_GATE}"
        node = analysis_runner(
            "02_virtual_plunger_calibration",
            ds_raw_all={pair_key: ds_raw},
            param_overrides=_default_param_overrides(),
            artifacts_subdir="02_virtual_plunger_qarray",
        )
        return node, pair_key, ds_raw, v_px, v_py, sensor_opt_mV

    def test_node_analysis_and_plot_actions(self, dot_model, analysis_runner):
        """Node run_actions produce fit results and diagnostic figures."""
        node, pair_key, _, _, _, _ = self._run_node_pipeline(dot_model, analysis_runner)

        assert "fit_results" in node.results, "analyse_data did not produce fit_results"
        assert pair_key in node.results["fit_results"]

        fit = node.results["fit_results"][pair_key]
        assert fit["plunger_gate_name"] == PLUNGER_X_GATE
        assert fit["device_gate_name"] == PLUNGER_Y_GATE
        assert fit["fit_params"]["success"], f"Virtual plunger fit failed: {fit}"

        T = fit["T_matrix"]
        assert T is not None
        assert T.shape == (2, 2)
        assert abs(np.linalg.det(T)) > 1e-6, f"T matrix is singular: det={np.linalg.det(T)}"

        assert pair_key in node.results.get("figures", {}), "plot_data did not produce a figure"
        fig = node.results["figures"][pair_key]

        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "02_virtual_plunger_qarray"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "transition_analysis.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "transition_analysis.png").exists()

    def test_plot_charge_stability(self, dot_model):
        """Generate charge-stability diagrams with and without sensor compensation."""
        sensor_opt_mV = self._find_sensor_opt(dot_model)

        v_px = sweep_voltages_mV(PLUNGER_X_CENTER_V, PLUNGER_X_SPAN_V, PLUNGER_X_POINTS)
        v_py = sweep_voltages_mV(PLUNGER_Y_CENTER_V, PLUNGER_Y_SPAN_V, PLUNGER_Y_POINTS)

        ds_raw_uncomp = simulate_plunger_plunger_scan(
            dot_model,
            v_px,
            v_py,
            plunger_x_gate_idx=0,
            plunger_y_gate_idx=1,
            sensor_operating_point=sensor_opt_mV,
        )
        ds_raw_comp = simulate_plunger_plunger_scan(
            dot_model,
            v_px,
            v_py,
            plunger_x_gate_idx=0,
            plunger_y_gate_idx=1,
            sensor_operating_point=sensor_opt_mV,
            sensor_compensation=SENSOR_COMP,
        )

        amp_uncomp = np.hypot(ds_raw_uncomp["I"].values[0], ds_raw_uncomp["Q"].values[0])
        amp_comp = np.hypot(ds_raw_comp["I"].values[0], ds_raw_comp["Q"].values[0])

        v_x = ds_raw_comp.coords["x_volts"].values
        v_y = ds_raw_comp.coords["y_volts"].values
        extent = [v_x[0], v_x[-1], v_y[0], v_y[-1]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(amp_uncomp, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[0].set_title("Without sensor compensation")
        axes[0].set_xlabel("Plunger 1 (V)")
        axes[0].set_ylabel("Plunger 2 (V)")

        axes[1].imshow(amp_comp, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[1].set_title("With sensor compensation")
        axes[1].set_xlabel("Plunger 1 (V)")
        axes[1].set_ylabel("Plunger 2 (V)")

        plt.tight_layout()
        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "02_virtual_plunger_qarray"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "charge_stability.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "charge_stability.png").exists()

    def test_virtual_gate_sweep(self, dot_model, analysis_runner):
        """Use node-derived T to sweep in virtual-gate space and compare maps."""
        node, pair_key, ds_raw, v_px, v_py, sensor_opt_mV = self._run_node_pipeline(dot_model, analysis_runner)

        fit = node.results["fit_results"][pair_key]
        assert fit["fit_params"]["success"], f"Virtual plunger fit failed: {fit}"
        T = fit["T_matrix"]
        assert T is not None

        raw_amplitude = np.hypot(ds_raw["I"].values[0], ds_raw["Q"].values[0])
        T_inv = np.linalg.inv(T)

        v_virt_x = v_px
        v_virt_y = v_py
        VX, VY = np.meshgrid(v_virt_x, v_virt_y)
        virt_flat = np.stack([VX.ravel(), VY.ravel()], axis=0)
        phys_flat = T_inv @ virt_flat

        phys_cx = T_inv @ np.array([(v_px[0] + v_px[-1]) / 2, (v_py[0] + v_py[-1]) / 2])

        base = np.zeros(7)
        alpha_x = SENSOR_COMP.get(0, 0.0)
        alpha_y = SENSOR_COMP.get(1, 0.0)

        voltage_array = np.tile(base, (phys_flat.shape[1], 1))
        voltage_array[:, 0] = phys_flat[0]
        voltage_array[:, 1] = phys_flat[1]
        voltage_array[:, 6] = (
            sensor_opt_mV + alpha_x * (phys_flat[0] - phys_cx[0]) + alpha_y * (phys_flat[1] - phys_cx[1])
        )

        z, _ = dot_model.charge_sensor_open(-voltage_array)
        virt_signal = z.squeeze().reshape(len(v_virt_y), len(v_virt_x))

        v_x_V = v_px * 1e-3
        v_y_V = v_py * 1e-3
        extent = [v_x_V[0], v_x_V[-1], v_y_V[0], v_y_V[-1]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(raw_amplitude, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[0].set_title("Raw plunger gates (with sensor comp)")
        axes[0].set_xlabel("Plunger 1 (V)")
        axes[0].set_ylabel("Plunger 2 (V)")

        axes[1].imshow(virt_signal, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[1].set_title("Virtual gates\n" f"T=[[{T[0,0]:.3f}, {T[0,1]:.3f}], [{T[1,0]:.3f}, {T[1,1]:.3f}]]")
        axes[1].set_xlabel("Virtual gate 1 (V)")
        axes[1].set_ylabel("Virtual gate 2 (V)")

        plt.tight_layout()
        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "02_virtual_plunger_qarray"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "virtual_gate_sweep.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "virtual_gate_sweep.png").exists()
