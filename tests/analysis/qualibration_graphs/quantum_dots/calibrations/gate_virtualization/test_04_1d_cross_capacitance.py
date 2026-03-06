"""Analysis tests for node 04 — 1D cross-capacitance measurement.

Tests the transition detection and cross-capacitance coefficient extraction
on both synthetic (analytic) data and qarray-simulated 1D plunger sweeps.

The synthetic tests verify that:
- detect_transition_position finds step-function transitions accurately
- extract_cross_capacitance_coefficient recovers known alpha values

The e2e tests use the qarray ChargeSensedDotArray to simulate paired 1D
sweeps with a perturbing gate offset, then run the node's full analysis
pipeline (analyse_data, plot_data, update_state) via ``analysis_runner``.

Gate names match the QuAM created by quam_factory:
  target plunger: virtual_dot_1
  perturbing gate: virtual_dot_2
  sensor: virtual_sensor_1
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from .conftest import (
    CALIBRATED_SENSOR_COMP,
    CALIBRATION_LIBRARY_ROOT,
    _call_node_action,
    _patch_action_manager_register_only,
    _reimport_node_to_register_actions,
    simulate_1d_plunger_sweep,
    simulate_sensor_sweep,
    sweep_voltages_mV,
)


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
_cc1d_mod = _load_module(
    "_gv_cc1d_analysis",
    _GATE_VIRT_UTILS / "cross_capacitance_1d_analysis.py",
)

process_raw_dataset = _analysis_mod.process_raw_dataset
detect_transition_position = _cc1d_mod.detect_transition_position
extract_cross_capacitance_coefficient = _cc1d_mod.extract_cross_capacitance_coefficient


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_step_signal(
    voltage: np.ndarray,
    transition_pos: float,
    width: float = 0.001,
    amplitude: float = 1.0,
    offset: float = 0.0,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a sigmoid step function signal at a given transition position."""
    signal = offset + amplitude / (1.0 + np.exp(-(voltage - transition_pos) / width))
    if noise_std > 0:
        if rng is None:
            rng = np.random.default_rng(42)
        signal = signal + rng.normal(0, noise_std, size=signal.shape)
    return signal


def _make_1d_dataset(
    voltage: np.ndarray,
    signal: np.ndarray,
) -> xr.Dataset:
    """Wrap voltage + signal into a Dataset matching the node's raw format."""
    I_data = signal[np.newaxis, :]
    Q_data = np.zeros_like(I_data)
    return xr.Dataset(
        {
            "I": xr.DataArray(
                I_data,
                dims=["sensors", "x_volts"],
                coords={"sensors": ["sensor_1"], "x_volts": voltage},
            ),
            "Q": xr.DataArray(
                Q_data,
                dims=["sensors", "x_volts"],
                coords={"sensors": ["sensor_1"], "x_volts": voltage},
            ),
        }
    )


# ── Synthetic tests ─────────────────────────────────────────────────────────


class TestDetectTransitionPosition:
    """Verify transition detection on clean and noisy step functions."""

    @pytest.mark.parametrize("true_pos", [-0.010, 0.0, 0.005, 0.015])
    def test_noiseless_step(self, true_pos):
        voltage = np.linspace(-0.025, 0.025, 201)
        signal = _make_step_signal(voltage, true_pos, width=0.001)
        result = detect_transition_position(voltage, signal, method="gradient")
        assert result["success"] or abs(result["position"] - true_pos) < 0.002
        assert abs(result["position"] - true_pos) < 0.002

    def test_noisy_step(self):
        rng = np.random.default_rng(99)
        voltage = np.linspace(-0.025, 0.025, 401)
        true_pos = 0.005
        signal = _make_step_signal(voltage, true_pos, width=0.001, noise_std=0.02, rng=rng)
        result = detect_transition_position(voltage, signal, method="gradient")
        assert abs(result["position"] - true_pos) < 0.005

    @pytest.mark.parametrize("method", ["gradient", "bayesian_cp"])
    def test_methods_agree(self, method):
        voltage = np.linspace(-0.025, 0.025, 201)
        true_pos = 0.003
        signal = _make_step_signal(voltage, true_pos, width=0.001)
        result = detect_transition_position(voltage, signal, method=method)
        assert abs(result["position"] - true_pos) < 0.003


class TestExtractCrossCapacitanceCoefficient:
    """Verify coefficient extraction on synthetic paired sweeps."""

    @pytest.mark.parametrize("true_alpha", [0.0, 0.1, -0.2, 0.5])
    def test_recover_alpha_noiseless(self, true_alpha):
        voltage = np.linspace(-0.025, 0.025, 201)
        step_voltage = 0.010
        ref_pos = 0.0
        shifted_pos = ref_pos + true_alpha * step_voltage

        sig_ref = _make_step_signal(voltage, ref_pos, width=0.001)
        sig_shifted = _make_step_signal(voltage, shifted_pos, width=0.001)

        ds_ref = process_raw_dataset(_make_1d_dataset(voltage, sig_ref))
        ds_shifted = process_raw_dataset(_make_1d_dataset(voltage, sig_shifted))

        result = extract_cross_capacitance_coefficient(
            ds_ref,
            ds_shifted,
            step_voltage=step_voltage,
            target_gate="virtual_dot_1",
            perturb_gate="virtual_dot_2",
            method="gradient",
        )

        assert result["target_gate"] == "virtual_dot_1"
        assert result["perturb_gate"] == "virtual_dot_2"
        assert abs(result["coefficient"] - true_alpha) < 0.05

    def test_recover_alpha_with_noise(self):
        rng = np.random.default_rng(42)
        voltage = np.linspace(-0.025, 0.025, 401)
        step_voltage = 0.010
        true_alpha = 0.15
        ref_pos = 0.0
        shifted_pos = ref_pos + true_alpha * step_voltage

        sig_ref = _make_step_signal(voltage, ref_pos, width=0.001, noise_std=0.03, rng=rng)
        sig_shifted = _make_step_signal(
            voltage,
            shifted_pos,
            width=0.001,
            noise_std=0.03,
            rng=np.random.default_rng(43),
        )

        ds_ref = process_raw_dataset(_make_1d_dataset(voltage, sig_ref))
        ds_shifted = process_raw_dataset(_make_1d_dataset(voltage, sig_shifted))

        result = extract_cross_capacitance_coefficient(
            ds_ref,
            ds_shifted,
            step_voltage=step_voltage,
            target_gate="virtual_dot_1",
            perturb_gate="virtual_dot_2",
            method="gradient",
        )

        assert abs(result["coefficient"] - true_alpha) < 0.1

    def test_zero_step_voltage_raises(self):
        voltage = np.linspace(-0.025, 0.025, 201)
        sig = _make_step_signal(voltage, 0.0, width=0.001)
        ds = process_raw_dataset(_make_1d_dataset(voltage, sig))

        with pytest.raises(ValueError, match="step_voltage must be non-zero"):
            extract_cross_capacitance_coefficient(
                ds,
                ds,
                step_voltage=0.0,
                target_gate="a",
                perturb_gate="b",
            )


# ── update_state unit tests (no qarray needed) ─────────────────────────────

TARGET_GATE = "virtual_dot_1"
PERTURB_GATE = "virtual_dot_2"
SENSOR_GATE = "virtual_sensor_1"

SWEEP_CENTER_V = 0.0
SWEEP_SPAN_V = 0.050
SWEEP_POINTS = 201
STEP_VOLTAGE_V = 0.010

SENSOR_SWEEP_CENTER_MV = 5.0
SENSOR_SWEEP_SPAN_MV = 6.0
SENSOR_SWEEP_POINTS = 300

SENSOR_COMP = CALIBRATED_SENSOR_COMP


def test_update_state_compose(minimal_quam_factory):
    """update_state in compose mode right-composes the correction into the full column."""
    from quam_config import Quam

    machine = minimal_quam_factory()
    layer = machine.virtual_gate_sets["main_qpu"].layers[0]
    source_gates = list(layer.source_gates)

    target_idx = source_gates.index(TARGET_GATE)
    perturb_idx = source_gates.index(PERTURB_GATE)

    full_old = np.asarray(layer.matrix, dtype=float).copy()

    # Seed some non-trivial off-diagonal values so we can verify
    # that the full-column update propagates correctly.
    layer.matrix[target_idx][perturb_idx] = 0.05
    full_old[target_idx, perturb_idx] = 0.05
    # Also seed a value in another row of the target column so we
    # can verify it propagates into the perturb column.
    other_row = (target_idx + 2) % len(source_gates)
    layer.matrix[other_row][target_idx] = 0.03
    full_old[other_row, target_idx] = 0.03

    measured_alpha = 0.12

    with (
        patch.object(Quam, "load", return_value=machine),
        _patch_action_manager_register_only(),
    ):
        node = _reimport_node_to_register_actions(
            "04_1d_cross_capacitance",
            CALIBRATION_LIBRARY_ROOT,
        )
    assert node is not None
    node.machine = machine
    node.parameters.update_mode = "compose"

    pair_key = f"{TARGET_GATE}_vs_{PERTURB_GATE}"
    node.results["fit_results"] = {
        pair_key: {
            "coefficient": measured_alpha,
            "target_gate": TARGET_GATE,
            "perturb_gate": PERTURB_GATE,
            "fit_params": {"success": True},
        }
    }

    _call_node_action(node, "update_state")

    updated = np.asarray(machine.virtual_gate_sets["main_qpu"].layers[0].matrix, dtype=float)
    cols = [target_idx, perturb_idx]
    delta = np.eye(2)
    delta[0, 1] = measured_alpha
    expected = full_old.copy()
    expected[:, cols] = full_old[:, cols] @ delta

    np.testing.assert_allclose(
        updated[:, cols],
        expected[:, cols],
        atol=1e-12,
        err_msg="compose mode should right-compose Delta into both columns",
    )
    # Verify the target column is unchanged.
    np.testing.assert_allclose(updated[:, target_idx], full_old[:, target_idx], atol=1e-12)
    # Verify the perturb column picked up contributions from the target column.
    assert updated[other_row][perturb_idx] == pytest.approx(
        full_old[other_row, perturb_idx] + measured_alpha * full_old[other_row, target_idx]
    )


def test_update_state_overwrite(minimal_quam_factory):
    """update_state in overwrite mode replaces the single entry with the measured value."""
    from quam_config import Quam

    machine = minimal_quam_factory()
    layer = machine.virtual_gate_sets["main_qpu"].layers[0]
    source_gates = list(layer.source_gates)

    target_idx = source_gates.index(TARGET_GATE)
    perturb_idx = source_gates.index(PERTURB_GATE)

    layer.matrix[target_idx][perturb_idx] = 0.99

    measured_alpha = 0.25

    with (
        patch.object(Quam, "load", return_value=machine),
        _patch_action_manager_register_only(),
    ):
        node = _reimport_node_to_register_actions(
            "04_1d_cross_capacitance",
            CALIBRATION_LIBRARY_ROOT,
        )
    assert node is not None
    node.machine = machine
    node.parameters.update_mode = "overwrite"

    pair_key = f"{TARGET_GATE}_vs_{PERTURB_GATE}"
    node.results["fit_results"] = {
        pair_key: {
            "coefficient": measured_alpha,
            "target_gate": TARGET_GATE,
            "perturb_gate": PERTURB_GATE,
            "fit_params": {"success": True},
        }
    }

    _call_node_action(node, "update_state")

    updated = np.asarray(machine.virtual_gate_sets["main_qpu"].layers[0].matrix, dtype=float)
    assert updated[target_idx][perturb_idx] == pytest.approx(measured_alpha)


def test_update_state_skips_failed_fit(minimal_quam_factory):
    """update_state does not modify the matrix when the fit failed."""
    from quam_config import Quam

    machine = minimal_quam_factory()
    layer = machine.virtual_gate_sets["main_qpu"].layers[0]
    source_gates = list(layer.source_gates)

    target_row = source_gates.index(TARGET_GATE)
    perturb_col = source_gates.index(PERTURB_GATE)

    original_value = layer.matrix[target_row][perturb_col]

    with (
        patch.object(Quam, "load", return_value=machine),
        _patch_action_manager_register_only(),
    ):
        node = _reimport_node_to_register_actions(
            "04_1d_cross_capacitance",
            CALIBRATION_LIBRARY_ROOT,
        )
    assert node is not None
    node.machine = machine

    pair_key = f"{TARGET_GATE}_vs_{PERTURB_GATE}"
    node.results["fit_results"] = {
        pair_key: {
            "coefficient": 0.5,
            "target_gate": TARGET_GATE,
            "perturb_gate": PERTURB_GATE,
            "fit_params": {"success": False, "reason": "test"},
        }
    }

    _call_node_action(node, "update_state")

    updated = machine.virtual_gate_sets["main_qpu"].layers[0].matrix
    assert updated[target_row][perturb_col] == pytest.approx(original_value)


# ── qarray e2e simulation tests ─────────────────────────────────────────────


def _qarray_available() -> bool:
    """Check whether qarray + JAX can actually run."""
    try:
        from qarray import DotArray

        m = DotArray(Cdd=[[0.1]], Cgd=[[0.1]], algorithm="default", implementation="jax")
        m.ground_state_open(np.array([[0.0], [0.1]]))
        return True
    except Exception:
        return False


def _default_param_overrides(**extra):
    """Merge test-default node parameters with any caller overrides."""
    base = {
        "cross_capacitance_mapping": {TARGET_GATE: [PERTURB_GATE]},
        "step_voltage": STEP_VOLTAGE_V,
        "sweep_span": SWEEP_SPAN_V,
        "sweep_points": SWEEP_POINTS,
    }
    base.update(extra)
    return base


@pytest.mark.analysis
@pytest.mark.skipif(not _qarray_available(), reason="qarray/JAX not functional in this environment")
class TestCrossCapacitance1DE2E:
    """End-to-end: qarray simulation -> node analyse_data/plot_data/update_state."""

    def _find_sensor_opt(self, dot_model) -> float:
        """Run a 1D sensor sweep and return the operating point in mV."""
        base_voltages = np.zeros(7)
        base_voltages[0] = SWEEP_CENTER_V * 1e3
        base_voltages[1] = SWEEP_CENTER_V * 1e3

        v_sensor_mV = np.linspace(
            SENSOR_SWEEP_CENTER_MV - SENSOR_SWEEP_SPAN_MV / 2,
            SENSOR_SWEEP_CENTER_MV + SENSOR_SWEEP_SPAN_MV / 2,
            SENSOR_SWEEP_POINTS,
        )
        ds = simulate_sensor_sweep(dot_model, v_sensor_mV, base_voltages=base_voltages)
        signal = np.hypot(ds["I"].values[0], ds["Q"].values[0])
        v_sensor_V = ds.coords["x_volts"].values
        return float(v_sensor_V[int(np.argmax(signal))] * 1e3)

    def _simulate_paired_sweeps(self, dot_model):
        """Simulate a reference and shifted 1D plunger sweep via qarray."""
        sensor_opt_mV = self._find_sensor_opt(dot_model)
        v_plunger = sweep_voltages_mV(SWEEP_CENTER_V, SWEEP_SPAN_V, SWEEP_POINTS)
        step_mV = STEP_VOLTAGE_V * 1e3

        ds_ref = simulate_1d_plunger_sweep(
            dot_model,
            v_plunger,
            plunger_gate_idx=0,
            sensor_operating_point=sensor_opt_mV,
            perturb_gate_idx=1,
            perturb_offset_mV=0.0,
            sensor_compensation=SENSOR_COMP,
        )
        ds_shifted = simulate_1d_plunger_sweep(
            dot_model,
            v_plunger,
            plunger_gate_idx=0,
            sensor_operating_point=sensor_opt_mV,
            perturb_gate_idx=1,
            perturb_offset_mV=step_mV,
            sensor_compensation=SENSOR_COMP,
        )
        return ds_ref, ds_shifted, sensor_opt_mV

    def _run_node_pipeline(self, dot_model, analysis_runner):
        """Simulate paired sweeps and run the full node 04 pipeline."""
        ds_ref, ds_shifted, sensor_opt_mV = self._simulate_paired_sweeps(dot_model)

        pair_key = f"{TARGET_GATE}_vs_{PERTURB_GATE}"
        ref_key = f"{pair_key}_ref"
        shifted_key = f"{pair_key}_shifted"

        node = analysis_runner(
            "04_1d_cross_capacitance",
            ds_raw_all={ref_key: ds_ref, shifted_key: ds_shifted},
            param_overrides=_default_param_overrides(),
            artifacts_subdir="04_1d_cross_capacitance_qarray",
        )
        return node, pair_key, ds_ref, ds_shifted

    def test_node_analysis_single_pair(self, dot_model, analysis_runner):
        """Node run_actions produce fit results with a finite coefficient."""
        node, pair_key, _, _ = self._run_node_pipeline(dot_model, analysis_runner)

        assert "fit_results" in node.results, "analyse_data did not produce fit_results"
        assert pair_key in node.results["fit_results"]

        fit = node.results["fit_results"][pair_key]
        assert fit["target_gate"] == TARGET_GATE
        assert fit["perturb_gate"] == PERTURB_GATE
        alpha = fit["coefficient"]
        assert np.isfinite(alpha), f"alpha is not finite: {alpha}"
        assert fit["pos_ref"] is not None
        assert fit["pos_shifted"] is not None

    def test_matrix_updated_after_pipeline(self, dot_model, analysis_runner):
        """update_state writes the coefficient into the compensation matrix."""
        node, pair_key, _, _ = self._run_node_pipeline(dot_model, analysis_runner)

        fit = node.results["fit_results"][pair_key]
        alpha = fit["coefficient"]

        layer = node.machine.virtual_gate_sets["main_qpu"].layers[0]
        source_gates = list(layer.source_gates)
        target_row = source_gates.index(TARGET_GATE)
        perturb_col = source_gates.index(PERTURB_GATE)

        # Default update_mode is "compose"; matrix starts as identity so
        # M_new[r][c] = M_old[r][c] + alpha * M_old[r][r] = 0 + alpha * 1 = alpha.
        assert layer.matrix[target_row][perturb_col] == pytest.approx(alpha), (
            f"Expected matrix[{target_row}][{perturb_col}] = {alpha}, " f"got {layer.matrix[target_row][perturb_col]}"
        )

    def test_plot_diagnostic(self, dot_model, analysis_runner):
        """plot_data produces a 2-panel diagnostic figure and saves as artifact."""
        node, pair_key, _, _ = self._run_node_pipeline(dot_model, analysis_runner)

        assert pair_key in node.results.get("figures", {}), "plot_data did not produce a figure"
        fig = node.results["figures"][pair_key]

        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "04_1d_cross_capacitance_qarray"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "diagnostic.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "diagnostic.png").exists()

    def test_plot_paired_traces(self, dot_model):
        """Generate a standalone comparison plot of reference vs shifted 1D traces."""
        sensor_opt_mV = self._find_sensor_opt(dot_model)
        v_plunger = sweep_voltages_mV(SWEEP_CENTER_V, SWEEP_SPAN_V, SWEEP_POINTS)
        step_mV = STEP_VOLTAGE_V * 1e3

        ds_ref = simulate_1d_plunger_sweep(
            dot_model,
            v_plunger,
            plunger_gate_idx=0,
            sensor_operating_point=sensor_opt_mV,
            perturb_gate_idx=1,
            perturb_offset_mV=0.0,
            sensor_compensation=SENSOR_COMP,
        )
        ds_shifted = simulate_1d_plunger_sweep(
            dot_model,
            v_plunger,
            plunger_gate_idx=0,
            sensor_operating_point=sensor_opt_mV,
            perturb_gate_idx=1,
            perturb_offset_mV=step_mV,
            sensor_compensation=SENSOR_COMP,
        )

        v_V = ds_ref.coords["x_volts"].values
        sig_ref = np.hypot(ds_ref["I"].values[0], ds_ref["Q"].values[0])
        sig_shifted = np.hypot(ds_shifted["I"].values[0], ds_shifted["Q"].values[0])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(v_V * 1e3, sig_ref, "b-", linewidth=1.2, label="Reference")
        ax.plot(v_V * 1e3, sig_shifted, color="orange", linewidth=1.2, label=f"Perturb +{STEP_VOLTAGE_V * 1e3:.0f} mV")
        ax.set_xlabel("Plunger voltage (mV)")
        ax.set_ylabel("Sensor signal (a.u.)")
        ax.set_title(f"1D cross-capacitance: {TARGET_GATE} vs {PERTURB_GATE}")
        ax.legend(fontsize=9)
        plt.tight_layout()

        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "04_1d_cross_capacitance_qarray"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "paired_traces.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "paired_traces.png").exists()
