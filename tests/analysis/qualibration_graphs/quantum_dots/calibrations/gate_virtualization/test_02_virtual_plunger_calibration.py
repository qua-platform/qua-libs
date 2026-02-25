"""Analysis tests for node 02 — virtual plunger calibration.

Uses the qarray ChargeSensedDotArray to simulate a 2D charge-stability
scan of two plunger gates, then runs the node's analysis pipeline
end-to-end.  The analysis detects charge transition lines via BayesianCP
edge detection and line fitting, extracts the two primary transition
angles, and constructs the virtual gate transformation matrix T = S @ R.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from .conftest import (
    CALIBRATED_SENSOR_COMP,
    simulate_plunger_plunger_scan,
    simulate_sensor_sweep,
    sweep_voltages_mV,
)

# Load analysis modules directly to avoid the package __init__.py
_GATE_VIRT_UTILS = (
    Path(__file__).resolve().parents[6]
    / "qualibration_graphs"
    / "quantum_dots"
    / "calibration_utils"
    / "gate_virtualization"
)


def _load_module(name: str, filepath: Path):
    spec = spec_from_file_location(name, filepath)
    mod = module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_analysis_mod = _load_module("_gv_analysis", _GATE_VIRT_UTILS / "analysis.py")
process_raw_dataset = _analysis_mod.process_raw_dataset

_sensor_analysis_mod = _load_module("_sd_analysis", _GATE_VIRT_UTILS / "sensor_dot_analysis.py")
fit_lorentzian = _sensor_analysis_mod.fit_lorentzian

_vp_analysis_mod = _load_module("_vp_analysis", _GATE_VIRT_UTILS / "virtual_plunger_analysis.py")
extract_virtual_plunger_coefficients = _vp_analysis_mod.extract_virtual_plunger_coefficients


# ── Test constants (matching node parameter conventions) ─────────────────────

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
    """Merge test-default scan parameters with any caller overrides."""
    base = {
        "x_center": PLUNGER_X_CENTER_V,
        "x_span": PLUNGER_X_SPAN_V,
        "x_points": PLUNGER_X_POINTS,
        "y_center": PLUNGER_Y_CENTER_V,
        "y_span": PLUNGER_Y_SPAN_V,
        "y_points": PLUNGER_Y_POINTS,
    }
    base.update(extra)
    return base


# ── qarray availability check ───────────────────────────────────────────────


def _qarray_available() -> bool:
    """Check whether qarray + JAX can actually run."""
    try:
        from qarray import DotArray

        m = DotArray(
            Cdd=[[0.1]], Cgd=[[0.1]], algorithm="default", implementation="jax"
        )
        m.ground_state_open(np.array([[0.0], [0.1]]))
        return True
    except Exception:
        return False


# ── e2e tests ────────────────────────────────────────────────────────────────


@pytest.mark.analysis
@pytest.mark.skipif(not _qarray_available(), reason="qarray/JAX not functional in this environment")
class TestVirtualPlungerE2E:
    """End-to-end: qarray simulation → inject ds_raw_all → node analysis pipeline."""

    def _find_sensor_opt(self, dot_model) -> float:
        """Run a 1D sensor sweep at the plunger center and return optimal point in mV."""
        base_voltages = np.zeros(7)
        base_voltages[0] = PLUNGER_X_CENTER_V * 1e3
        base_voltages[1] = PLUNGER_Y_CENTER_V * 1e3

        v_sensor_mV = np.linspace(
            SENSOR_SWEEP_CENTER_MV - SENSOR_SWEEP_SPAN_MV / 2,
            SENSOR_SWEEP_CENTER_MV + SENSOR_SWEEP_SPAN_MV / 2,
            SENSOR_SWEEP_POINTS,
        )
        ds = simulate_sensor_sweep(
            dot_model, v_sensor_mV, base_voltages=base_voltages,
        )
        signal = np.sqrt(ds["I"].values[0] ** 2 + ds["Q"].values[0] ** 2)
        v_V = ds.coords["x_volts"].values
        result = fit_lorentzian(v_V, signal, side="right")
        return result.optimal_voltage * 1e3

    def test_plot_charge_stability(self, dot_model):
        """Generate charge-stability diagrams with and without sensor compensation."""
        sensor_opt_mV = self._find_sensor_opt(dot_model)

        v_px = sweep_voltages_mV(PLUNGER_X_CENTER_V, PLUNGER_X_SPAN_V, PLUNGER_X_POINTS)
        v_py = sweep_voltages_mV(PLUNGER_Y_CENTER_V, PLUNGER_Y_SPAN_V, PLUNGER_Y_POINTS)

        ds_raw_uncomp = simulate_plunger_plunger_scan(
            dot_model, v_px, v_py,
            plunger_x_gate_idx=0, plunger_y_gate_idx=1,
            sensor_operating_point=sensor_opt_mV,
        )
        ds_raw_comp = simulate_plunger_plunger_scan(
            dot_model, v_px, v_py,
            plunger_x_gate_idx=0, plunger_y_gate_idx=1,
            sensor_operating_point=sensor_opt_mV,
            sensor_compensation=SENSOR_COMP,
        )

        ds_uncomp = process_raw_dataset(ds_raw_uncomp)
        ds_comp = process_raw_dataset(ds_raw_comp)

        v_x = ds_comp["amplitude"].coords["x_volts"].values
        v_y = ds_comp["amplitude"].coords["y_volts"].values
        extent = [v_x[0], v_x[-1], v_y[0], v_y[-1]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(
            ds_uncomp["amplitude"].isel(sensors=0).values,
            extent=extent, origin="lower", aspect="auto", cmap="hot",
        )
        axes[0].set_title("Without sensor compensation")
        axes[0].set_xlabel("Plunger 1 (V)")
        axes[0].set_ylabel("Plunger 2 (V)")

        axes[1].imshow(
            ds_comp["amplitude"].isel(sensors=0).values,
            extent=extent, origin="lower", aspect="auto", cmap="hot",
        )
        axes[1].set_title("With sensor compensation")
        axes[1].set_xlabel("Plunger 1 (V)")
        axes[1].set_ylabel("Plunger 2 (V)")

        plt.tight_layout()

        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "02_virtual_plunger_qarray"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "charge_stability.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "charge_stability.png").exists()

    def test_extract_transition_angles(self, dot_model):
        """Run full analysis pipeline: edge detect -> line fit -> T matrix."""
        sensor_opt_mV = self._find_sensor_opt(dot_model)

        v_px = sweep_voltages_mV(PLUNGER_X_CENTER_V, PLUNGER_X_SPAN_V, PLUNGER_X_POINTS)
        v_py = sweep_voltages_mV(PLUNGER_Y_CENTER_V, PLUNGER_Y_SPAN_V, PLUNGER_Y_POINTS)

        ds_raw = simulate_plunger_plunger_scan(
            dot_model, v_px, v_py,
            plunger_x_gate_idx=0, plunger_y_gate_idx=1,
            sensor_operating_point=sensor_opt_mV,
            sensor_compensation=SENSOR_COMP,
        )
        ds = process_raw_dataset(ds_raw)

        result = extract_virtual_plunger_coefficients(
            ds, plunger_x_name="x_volts", plunger_y_name="y_volts",
        )

        assert result is not None, "Analysis returned None"
        assert result["fit_params"]["success"], f"Fit failed: {result['fit_params']}"
        assert result["theta1"] is not None
        assert result["theta2"] is not None

        T = result["T_matrix"]
        assert T is not None
        assert T.shape == (2, 2)
        assert abs(np.linalg.det(T)) > 1e-6, f"T matrix is singular: det={np.linalg.det(T)}"

        # ── diagnostic plot ──────────────────────────────────────────────
        amplitude = ds["amplitude"].isel(sensors=0).values
        v_x = ds["amplitude"].coords["x_volts"].values
        v_y = ds["amplitude"].coords["y_volts"].values
        extent = [v_x[0], v_x[-1], v_y[0], v_y[-1]]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(amplitude, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[0].set_title("Compensated amplitude")
        axes[0].set_xlabel("Plunger 1 (V)")
        axes[0].set_ylabel("Plunger 2 (V)")

        mean_cp = result["mean_cp"]
        axes[1].imshow(mean_cp, origin="lower", aspect="auto", cmap="magma")
        axes[1].set_title("Edge map (BayesianCP)")

        ny, nx = amplitude.shape
        axes[2].imshow(amplitude, extent=extent, origin="lower", aspect="auto", cmap="hot")
        for seg in result["segments"]:
            r_start, c_start = seg.start
            r_end, c_end = seg.end
            x_s = v_x[0] + (c_start / nx) * (v_x[-1] - v_x[0])
            x_e = v_x[0] + (c_end / nx) * (v_x[-1] - v_x[0])
            y_s = v_y[0] + (r_start / ny) * (v_y[-1] - v_y[0])
            y_e = v_y[0] + (r_end / ny) * (v_y[-1] - v_y[0])
            axes[2].plot([x_s, x_e], [y_s, y_e], "c-", linewidth=1.5, alpha=0.8)
        theta1_deg = np.degrees(result["theta1"])
        theta2_deg = np.degrees(result["theta2"])
        axes[2].set_title(
            f"Segments + angles\n"
            f"θ1={theta1_deg:.1f}°  θ2={theta2_deg:.1f}°\n"
            f"T=[[{T[0,0]:.3f}, {T[0,1]:.3f}], [{T[1,0]:.3f}, {T[1,1]:.3f}]]"
        )
        axes[2].set_xlabel("Plunger 1 (V)")
        axes[2].set_ylabel("Plunger 2 (V)")

        plt.tight_layout()
        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "02_virtual_plunger_qarray"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "transition_analysis.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "transition_analysis.png").exists()

    def test_virtual_gate_sweep(self, dot_model):
        """Sweep virtual gates (via T^-1) and compare with raw plunger scan."""
        sensor_opt_mV = self._find_sensor_opt(dot_model)

        v_px = sweep_voltages_mV(PLUNGER_X_CENTER_V, PLUNGER_X_SPAN_V, PLUNGER_X_POINTS)
        v_py = sweep_voltages_mV(PLUNGER_Y_CENTER_V, PLUNGER_Y_SPAN_V, PLUNGER_Y_POINTS)

        # Raw plunger scan (with sensor compensation)
        ds_raw = simulate_plunger_plunger_scan(
            dot_model, v_px, v_py,
            plunger_x_gate_idx=0, plunger_y_gate_idx=1,
            sensor_operating_point=sensor_opt_mV,
            sensor_compensation=SENSOR_COMP,
        )
        ds = process_raw_dataset(ds_raw)
        raw_amplitude = ds["amplitude"].isel(sensors=0).values

        # Extract T matrix
        result = extract_virtual_plunger_coefficients(
            ds, plunger_x_name="x_volts", plunger_y_name="y_volts",
        )
        T = result["T_matrix"]
        T_inv = np.linalg.inv(T)

        # Sweep the same grid in virtual-gate space
        v_virt_x = v_px
        v_virt_y = v_py
        VX, VY = np.meshgrid(v_virt_x, v_virt_y)
        virt_flat = np.stack([VX.ravel(), VY.ravel()], axis=0)  # (2, N)
        phys_flat = T_inv @ virt_flat  # (2, N)

        phys_cx = T_inv @ np.array([(v_px[0] + v_px[-1]) / 2,
                                     (v_py[0] + v_py[-1]) / 2])

        n_gates = 7
        base = np.zeros(n_gates)
        alpha_x = SENSOR_COMP.get(0, 0.0)
        alpha_y = SENSOR_COMP.get(1, 0.0)

        voltage_array = np.tile(base, (phys_flat.shape[1], 1))
        voltage_array[:, 0] = phys_flat[0]
        voltage_array[:, 1] = phys_flat[1]
        voltage_array[:, 6] = (
            sensor_opt_mV
            + alpha_x * (phys_flat[0] - phys_cx[0])
            + alpha_y * (phys_flat[1] - phys_cx[1])
        )

        z, _ = dot_model.charge_sensor_open(-voltage_array)
        z = z.squeeze()
        virt_signal = z.reshape(len(v_virt_y), len(v_virt_x))

        # ── comparison plot ───────────────────────────────────────────────
        v_x_V = v_px * 1e-3
        v_y_V = v_py * 1e-3
        extent = [v_x_V[0], v_x_V[-1], v_y_V[0], v_y_V[-1]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(
            raw_amplitude, extent=extent, origin="lower",
            aspect="auto", cmap="hot",
        )
        axes[0].set_title("Raw plunger gates (with sensor comp)")
        axes[0].set_xlabel("Plunger 1 (V)")
        axes[0].set_ylabel("Plunger 2 (V)")

        axes[1].imshow(
            virt_signal, extent=extent, origin="lower",
            aspect="auto", cmap="hot",
        )
        axes[1].set_title(
            "Virtual gates\n"
            f"T=[[{T[0,0]:.3f}, {T[0,1]:.3f}], "
            f"[{T[1,0]:.3f}, {T[1,1]:.3f}]]"
        )
        axes[1].set_xlabel("Virtual gate 1 (V)")
        axes[1].set_ylabel("Virtual gate 2 (V)")

        plt.tight_layout()
        artifacts_dir = (
            Path(__file__).resolve().parents[4]
            / "artifacts" / "02_virtual_plunger_qarray"
        )
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "virtual_gate_sweep.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "virtual_gate_sweep.png").exists()
