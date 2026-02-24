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

from .conftest import simulate_plunger_plunger_scan, sweep_voltages_mV

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


# ── Test constants (matching node parameter conventions) ─────────────────────

PLUNGER_X_CENTER_V = 0.005
PLUNGER_X_SPAN_V = 0.010
PLUNGER_Y_CENTER_V = 0.005
PLUNGER_Y_SPAN_V = 0.010
PLUNGER_X_POINTS = 80
PLUNGER_Y_POINTS = 80


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

    def test_node_pipeline_single_pair(self, dot_model, analysis_runner):
        """Run the full pipeline with one plunger-plunger pair.

        Since the analysis function is a stub, fit_results entries will
        be None.  This test validates the wiring: the node loads, data
        is injected, analyse_data runs without error, and
        update_virtual_gate_matrix gracefully handles None results.
        """
        v_px = sweep_voltages_mV(PLUNGER_X_CENTER_V, PLUNGER_X_SPAN_V, PLUNGER_X_POINTS)
        v_py = sweep_voltages_mV(PLUNGER_Y_CENTER_V, PLUNGER_Y_SPAN_V, PLUNGER_Y_POINTS)

        ds_raw = simulate_plunger_plunger_scan(
            dot_model, v_px, v_py,
            plunger_x_gate_idx=0, plunger_y_gate_idx=1,
        )

        pair_key = "virtual_dot_1_vs_virtual_dot_2"
        node = analysis_runner(
            "02_virtual_plunger_calibration",
            ds_raw_all={pair_key: ds_raw},
            param_overrides=_default_param_overrides(
                plunger_device_mapping={"virtual_dot_1": ["virtual_dot_2"]},
            ),
            artifacts_subdir="02_virtual_plunger_e2e",
        )

        assert "fit_results" in node.results
        assert pair_key in node.results["fit_results"]
        # Stub returns None — once analysis is implemented, add value checks here
        assert node.results["fit_results"][pair_key] is None

    def test_node_pipeline_two_pairs(self, dot_model, analysis_runner):
        """Run with two plunger-plunger pairs."""
        v_px = sweep_voltages_mV(PLUNGER_X_CENTER_V, PLUNGER_X_SPAN_V, 60)
        v_py = sweep_voltages_mV(PLUNGER_Y_CENTER_V, PLUNGER_Y_SPAN_V, 60)

        ds_12 = simulate_plunger_plunger_scan(
            dot_model, v_px, v_py,
            plunger_x_gate_idx=0, plunger_y_gate_idx=1,
        )
        ds_13 = simulate_plunger_plunger_scan(
            dot_model, v_px, v_py,
            plunger_x_gate_idx=0, plunger_y_gate_idx=2,
        )

        node = analysis_runner(
            "02_virtual_plunger_calibration",
            ds_raw_all={
                "virtual_dot_1_vs_virtual_dot_2": ds_12,
                "virtual_dot_1_vs_virtual_dot_3": ds_13,
            },
            param_overrides=_default_param_overrides(
                x_points=60,
                y_points=60,
                plunger_device_mapping={
                    "virtual_dot_1": ["virtual_dot_2", "virtual_dot_3"],
                },
            ),
            artifacts_subdir="02_virtual_plunger_two_pairs",
        )

        assert len(node.results["fit_results"]) == 2

    def test_plot_charge_stability(self, dot_model):
        """Generate and save a charge-stability diagram from the qarray model."""
        v_px = sweep_voltages_mV(PLUNGER_X_CENTER_V, PLUNGER_X_SPAN_V, PLUNGER_X_POINTS)
        v_py = sweep_voltages_mV(PLUNGER_Y_CENTER_V, PLUNGER_Y_SPAN_V, PLUNGER_Y_POINTS)

        ds_raw = simulate_plunger_plunger_scan(
            dot_model, v_px, v_py,
            plunger_x_gate_idx=0, plunger_y_gate_idx=1,
        )
        ds = process_raw_dataset(ds_raw)

        v_x = ds["amplitude"].coords["x_volts"].values
        v_y = ds["amplitude"].coords["y_volts"].values
        amplitude = ds["amplitude"].isel(sensors=0).values
        extent = [v_x[0], v_x[-1], v_y[0], v_y[-1]]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(amplitude, extent=extent, origin="lower", aspect="auto", cmap="hot")
        ax.set_title("Charge stability diagram (plunger 1 vs plunger 2)")
        ax.set_xlabel("Plunger 1 (V)")
        ax.set_ylabel("Plunger 2 (V)")
        plt.tight_layout()

        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "02_virtual_plunger_qarray"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "charge_stability.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "charge_stability.png").exists()
