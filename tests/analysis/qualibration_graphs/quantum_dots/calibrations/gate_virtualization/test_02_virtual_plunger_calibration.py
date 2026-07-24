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
    apply_param_overrides,
    apply_sensor_noise,
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

        model = DotArray(
            Cdd=[[0.1]], Cgd=[[0.1]], algorithm="default", implementation="jax"
        )
        model.ground_state_open(np.array([[0.0], [0.1]]))
        return True
    except Exception:
        return False


def test_update_state_sets_off_diagonal_entries(minimal_quam_factory):
    """update_state additively sets the off-diagonal cross-talk entries from M."""
    from quam_config import Quam

    machine = minimal_quam_factory()
    layer = machine.virtual_gate_sets["main_qpu"].layers[0]
    source_gates = list(layer.source_gates)
    sensor_gate = "virtual_sensor_1"

    sensor_idx = source_gates.index(sensor_gate)
    plunger_idx = source_gates.index(PLUNGER_X_GATE)
    device_idx = source_gates.index(PLUNGER_Y_GATE)

    old = np.asarray(layer.matrix, dtype=float)
    old[sensor_idx, plunger_idx] = -0.020
    old[sensor_idx, device_idx] = -0.035
    old[plunger_idx, plunger_idx] = 1.0
    old[plunger_idx, device_idx] = 0.08
    old[device_idx, plunger_idx] = -0.06
    old[device_idx, device_idx] = 1.0
    layer.matrix = old.tolist()

    M = np.array(
        [
            [1.0, 0.14],
            [-0.09, 1.0],
        ],
        dtype=float,
    )
    expected = old.copy()
    expected[plunger_idx, device_idx] += M[0, 1]  # += 0.14
    expected[device_idx, plunger_idx] += M[1, 0]  # += -0.09

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
            "T_matrix": M,
        }
    }

    _call_node_action(node, "update_state")

    updated = np.asarray(
        machine.virtual_gate_sets["main_qpu"].layers[0].matrix,
        dtype=float,
    )
    # Only the two off-diagonal entries should change
    np.testing.assert_allclose(updated, expected, atol=1e-12)

    # Explicitly verify sensor row is unchanged
    np.testing.assert_allclose(updated[sensor_idx, :], old[sensor_idx, :], atol=1e-12)

    # Verify diagonals are unchanged
    assert updated[plunger_idx, plunger_idx] == old[plunger_idx, plunger_idx]
    assert updated[device_idx, device_idx] == old[device_idx, device_idx]


def test_update_state_asymmetric_for_non_plunger_pairs(minimal_quam_factory):
    """When plunger_gates is set and one gate is not a plunger, only the
    non-plunger→plunger entry is written (barrier, sensor, etc.)."""
    from quam_config import Quam

    machine = minimal_quam_factory()
    layer = machine.virtual_gate_sets["main_qpu"].layers[0]
    source_gates = list(layer.source_gates)

    plunger_gate = PLUNGER_X_GATE
    barrier_gate = "virtual_barrier_1"
    sensor_gate = "virtual_sensor_1"
    plunger_idx = source_gates.index(plunger_gate)
    barrier_idx = source_gates.index(barrier_gate)
    sensor_idx = source_gates.index(sensor_gate)

    old = np.asarray(layer.matrix, dtype=float)
    old[plunger_idx, barrier_idx] = 0.05
    old[barrier_idx, plunger_idx] = -0.03
    old[plunger_idx, sensor_idx] = 0.02
    old[sensor_idx, plunger_idx] = -0.01
    layer.matrix = old.tolist()

    M_barrier = np.array([[1.0, -0.35], [0.22, 1.0]], dtype=float)
    M_sensor = np.array([[1.0, 0.08], [-0.14, 1.0]], dtype=float)

    plunger_gates_list = [PLUNGER_X_GATE, PLUNGER_Y_GATE]

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
    apply_param_overrides(node, {"plunger_gates": plunger_gates_list})
    node.results["fit_results"] = {
        f"{plunger_gate}_vs_{barrier_gate}": {
            "fit_params": {"success": True},
            "T_matrix": M_barrier,
        },
        f"{plunger_gate}_vs_{sensor_gate}": {
            "fit_params": {"success": True},
            "T_matrix": M_sensor,
        },
    }

    _call_node_action(node, "update_state")

    updated = np.asarray(layer.matrix, dtype=float)

    # barrier→plunger entry should be updated (additive)
    assert updated[plunger_idx, barrier_idx] == pytest.approx(
        old[plunger_idx, barrier_idx] + M_barrier[0, 1]
    )
    # plunger→barrier entry should NOT be updated (asymmetric)
    assert updated[barrier_idx, plunger_idx] == pytest.approx(
        old[barrier_idx, plunger_idx]
    ), "Barrier row should not be modified by plunger-barrier pair"

    # sensor→plunger entry should be updated (additive)
    assert updated[plunger_idx, sensor_idx] == pytest.approx(
        old[plunger_idx, sensor_idx] + M_sensor[0, 1]
    )
    # plunger→sensor entry should NOT be updated (asymmetric)
    assert updated[sensor_idx, plunger_idx] == pytest.approx(
        old[sensor_idx, plunger_idx]
    ), "Sensor row should not be modified by plunger-sensor pair"


@pytest.mark.analysis
@pytest.mark.skipif(
    not _qarray_available(), reason="qarray/JAX not functional in this environment"
)
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
        assert (
            abs(np.linalg.det(T)) > 1e-6
        ), f"T matrix is singular: det={np.linalg.det(T)}"

        assert pair_key in node.results.get(
            "figures", {}
        ), "plot_data did not produce a figure"

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

        amp_uncomp = np.hypot(
            ds_raw_uncomp["I"].values[0], ds_raw_uncomp["Q"].values[0]
        )
        amp_comp = np.hypot(ds_raw_comp["I"].values[0], ds_raw_comp["Q"].values[0])

        v_x = ds_raw_comp.coords["x_volts"].values
        v_y = ds_raw_comp.coords["y_volts"].values
        extent = [v_x[0], v_x[-1], v_y[0], v_y[-1]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(
            amp_uncomp, extent=extent, origin="lower", aspect="auto", cmap="hot"
        )
        axes[0].set_xlim(v_x[0], v_x[-1])
        axes[0].set_ylim(v_y[0], v_y[-1])
        axes[0].set_title("Without sensor compensation")
        axes[0].set_xlabel("Plunger 1 (V)")
        axes[0].set_ylabel("Plunger 2 (V)")

        axes[1].imshow(
            amp_comp, extent=extent, origin="lower", aspect="auto", cmap="hot"
        )
        axes[1].set_xlim(v_x[0], v_x[-1])
        axes[1].set_ylim(v_y[0], v_y[-1])
        axes[1].set_title("With sensor compensation")
        axes[1].set_xlabel("Plunger 1 (V)")
        axes[1].set_ylabel("Plunger 2 (V)")

        plt.tight_layout()
        artifacts_dir = (
            Path(__file__).resolve().parents[4]
            / "artifacts"
            / "02_virtual_plunger_qarray"
        )
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "charge_stability.png", dpi=150)
        fig.savefig(artifacts_dir / "charge_stability.svg")
        plt.close(fig)
        assert (artifacts_dir / "charge_stability.png").exists()

    def test_virtual_gate_sweep(self, dot_model, analysis_runner):
        """Run transition analysis on both physical and virtualized plunger sweeps."""
        import xarray as xr

        # 1. Run node analysis on the physical (pre-virtualized) scan
        node_phys, pair_key, ds_raw, v_px, v_py, sensor_opt_mV = (
            self._run_node_pipeline(
                dot_model,
                analysis_runner,
            )
        )

        fit = node_phys.results["fit_results"][pair_key]
        assert fit["fit_params"]["success"], f"Virtual plunger fit failed: {fit}"
        M = fit["T_matrix"]
        assert M is not None

        # Save the physical transition analysis figure
        artifacts_dir = (
            Path(__file__).resolve().parents[4]
            / "artifacts"
            / "02_virtual_plunger_qarray"
        )
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if pair_key in node_phys.results.get("figures", {}):
            fig_phys = node_phys.results["figures"][pair_key]
            fig_phys.savefig(artifacts_dir / "transition_analysis_physical.png", dpi=150)
            fig_phys.savefig(artifacts_dir / "transition_analysis_physical.svg")
            plt.close(fig_phys)

        # 2. Build the virtualized scan dataset
        # M is virtual→physical; apply it directly to virtual sweep coordinates
        VX, VY = np.meshgrid(v_px, v_py)
        virt_flat = np.stack([VX.ravel(), VY.ravel()], axis=0)
        phys_flat = M @ virt_flat
        phys_cx = M @ np.array([(v_px[0] + v_px[-1]) / 2, (v_py[0] + v_py[-1]) / 2])

        alpha_x = SENSOR_COMP.get(0, 0.0)
        alpha_y = SENSOR_COMP.get(1, 0.0)

        voltage_array = np.zeros((phys_flat.shape[1], 7))
        voltage_array[:, 0] = phys_flat[0]
        voltage_array[:, 1] = phys_flat[1]
        voltage_array[:, 6] = (
            sensor_opt_mV
            + alpha_x * (phys_flat[0] - phys_cx[0])
            + alpha_y * (phys_flat[1] - phys_cx[1])
        )

        z, _ = dot_model.charge_sensor_open(-voltage_array)
        virt_signal = np.asarray(z).squeeze().reshape(len(v_py), len(v_px))
        virt_signal = apply_sensor_noise(virt_signal)

        v_x_V = v_px * 1e-3
        v_y_V = v_py * 1e-3
        ds_virt = xr.Dataset(
            {
                "I": xr.DataArray(
                    virt_signal[np.newaxis, :, :],
                    dims=["sensors", "y_volts", "x_volts"],
                    coords={
                        "sensors": ["sensor_1"],
                        "x_volts": v_x_V,
                        "y_volts": v_y_V,
                    },
                ),
                "Q": xr.DataArray(
                    np.zeros_like(virt_signal)[np.newaxis, :, :],
                    dims=["sensors", "y_volts", "x_volts"],
                    coords={
                        "sensors": ["sensor_1"],
                        "x_volts": v_x_V,
                        "y_volts": v_y_V,
                    },
                ),
            }
        )

        # 3. Run node analysis on the virtualized scan
        virt_pair_key = f"{PLUNGER_X_GATE}_vs_{PLUNGER_Y_GATE}"
        node_virt = analysis_runner(
            "02_virtual_plunger_calibration",
            ds_raw_all={virt_pair_key: ds_virt},
            param_overrides=_default_param_overrides(),
            artifacts_subdir="02_virtual_plunger_qarray",
        )

        # Save the virtualized transition analysis figure
        if virt_pair_key in node_virt.results.get("figures", {}):
            fig_virt = node_virt.results["figures"][virt_pair_key]
            fig_virt.savefig(artifacts_dir / "transition_analysis_virtual.png", dpi=150)
            fig_virt.savefig(artifacts_dir / "transition_analysis_virtual.svg")
            plt.close(fig_virt)

        # 4. Side-by-side comparison plot
        raw_amplitude = np.hypot(ds_raw["I"].values[0], ds_raw["Q"].values[0])
        extent = [v_x_V[0], v_x_V[-1], v_y_V[0], v_y_V[-1]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(
            raw_amplitude, extent=extent, origin="lower", aspect="auto", cmap="hot"
        )
        axes[0].set_xlim(v_x_V[0], v_x_V[-1])
        axes[0].set_ylim(v_y_V[0], v_y_V[-1])
        axes[0].set_title("Physical plunger gates (sensor-compensated)")
        axes[0].set_xlabel("Plunger 1 (V)")
        axes[0].set_ylabel("Plunger 2 (V)")

        axes[1].imshow(
            virt_signal, extent=extent, origin="lower", aspect="auto", cmap="hot"
        )
        axes[1].set_xlim(v_x_V[0], v_x_V[-1])
        axes[1].set_ylim(v_y_V[0], v_y_V[-1])
        axes[1].set_title(
            "Virtual gates (plunger-virtualized)\n"
            f"M=[[{M[0,0]:.3f}, {M[0,1]:.3f}], [{M[1,0]:.3f}, {M[1,1]:.3f}]]"
        )
        axes[1].set_xlabel("Virtual gate 1 (V)")
        axes[1].set_ylabel("Virtual gate 2 (V)")

        plt.tight_layout()
        fig.savefig(artifacts_dir / "virtual_gate_sweep.png", dpi=150)
        fig.savefig(artifacts_dir / "virtual_gate_sweep.svg")
        plt.close(fig)
        assert (artifacts_dir / "virtual_gate_sweep.png").exists()
