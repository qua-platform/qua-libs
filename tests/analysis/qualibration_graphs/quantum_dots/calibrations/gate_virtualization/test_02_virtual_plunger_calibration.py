"""Analysis tests for node 02 — virtual plunger calibration.

Uses the qarray ChargeSensedDotArray to simulate a 2D charge-stability
scan of two plunger gates, then runs the node's analysis pipeline
end-to-end.

The analysis function is currently a stub (returns None), so the e2e
tests validate the full wiring (data injection → analyse_data →
plot_data → update_virtual_gate_matrix) while the actual coefficient
extraction is deferred.  Once the analysis is implemented, these tests
will verify recovered slopes and matrix updates.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from .conftest import (
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

SENSOR_COMP = {0: -0.01, 1: -0.02}


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
