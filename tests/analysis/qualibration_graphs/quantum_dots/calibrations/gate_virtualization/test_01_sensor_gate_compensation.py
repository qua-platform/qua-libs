"""Analysis tests for node 01 — sensor gate compensation.

Uses the qarray ChargeSensedDotArray to simulate a 2D scan of the sensor
gate vs a device gate, then runs the node's full analysis pipeline
(analyse_data, plot_data, update_virtual_gate_matrix) end-to-end.

The sensor Coulomb peak shifts linearly with the device gate voltage;
fitting a shifted Lorentzian recovers the cross-talk coefficient (alpha).
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from .conftest import simulate_sensor_device_scan, sweep_voltages_mV

# Load analysis modules directly to avoid the package __init__.py, which
# transitively pulls in optional UI dependencies at import time.
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
_sensor_mod = _load_module(
    "_gv_sensor_analysis",
    _GATE_VIRT_UTILS / "sensor_compensation_analysis.py",
)

process_raw_dataset = _analysis_mod.process_raw_dataset
extract_sensor_compensation_coefficients = _sensor_mod.extract_sensor_compensation_coefficients
fit_shifted_lorentzian = _sensor_mod.fit_shifted_lorentzian
shifted_lorentzian_2d = _sensor_mod.shifted_lorentzian_2d


# ── Synthetic (analytic) Lorentzian tests ────────────────────────────────────


class TestShiftedLorentzian2D:
    """Verify the model function and fitter on exact synthetic data."""

    def test_model_shape(self):
        v_s = np.linspace(-1, 1, 50)
        v_d = np.linspace(-0.5, 0.5, 30)
        z = shifted_lorentzian_2d(v_s, v_d, A=1.0, v0=0.0, alpha=0.1, gamma=0.2, offset=0.0)
        assert z.shape == (30, 50)

    def test_peak_position_at_zero_device(self):
        v_s = np.linspace(-2, 2, 500)
        v_d = np.array([0.0])
        z = shifted_lorentzian_2d(v_s, v_d, A=1.0, v0=0.3, alpha=0.1, gamma=0.1, offset=0.0)
        peak_idx = np.argmax(z[0])
        assert abs(v_s[peak_idx] - 0.3) < (v_s[1] - v_s[0])

    def test_peak_shifts_with_device_gate(self):
        v_s = np.linspace(-5, 5, 1000)
        alpha = 0.5
        v0 = 0.0
        for vd_val in [-1.0, 0.0, 1.0, 2.0]:
            v_d = np.array([vd_val])
            z = shifted_lorentzian_2d(v_s, v_d, A=1.0, v0=v0, alpha=alpha, gamma=0.2, offset=0.0)
            expected_peak = v0 + alpha * vd_val
            peak_idx = np.argmax(z[0])
            assert abs(v_s[peak_idx] - expected_peak) < (v_s[1] - v_s[0]) * 2


class TestFitShiftedLorentzian:
    """Round-trip: generate data from known params, fit, recover params."""

    @pytest.mark.parametrize(
        "true_alpha",
        [0.0, 0.05, -0.1, 0.3],
    )
    def test_recover_alpha_noiseless(self, true_alpha):
        v_s = np.linspace(-2, 2, 100)
        v_d = np.linspace(-1, 1, 60)
        true_params = dict(A=1.0, v0=0.0, alpha=true_alpha, gamma=0.3, offset=0.5)
        signal = shifted_lorentzian_2d(v_s, v_d, **true_params)

        result = fit_shifted_lorentzian(v_s, v_d, signal)
        assert result["success"]
        assert abs(result["alpha"] - true_alpha) < 1e-3

    def test_recover_alpha_with_noise(self):
        rng = np.random.default_rng(42)
        v_s = np.linspace(-3, 3, 200)
        v_d = np.linspace(-1, 1, 120)
        true_alpha = 0.15
        signal = shifted_lorentzian_2d(v_s, v_d, A=2.0, v0=0.5, alpha=true_alpha, gamma=0.4, offset=1.0)
        noise = rng.normal(0, 0.05, size=signal.shape)
        signal_noisy = signal + noise

        result = fit_shifted_lorentzian(v_s, v_d, signal_noisy, maxiter=2000)
        assert result["success"]
        assert abs(result["alpha"] - true_alpha) < 0.05


# ── qarray e2e simulation tests ─────────────────────────────────────────────


def _qarray_available() -> bool:
    """Check whether qarray + JAX can actually run (catches version mismatches)."""
    try:
        from qarray import DotArray

        m = DotArray(
            Cdd=[[0.1]], Cgd=[[0.1]], algorithm="default", implementation="jax"
        )
        m.ground_state_open(np.array([[0.0], [0.1]]))
        return True
    except Exception:
        return False


SENSOR_CENTER_V = 0.015
SENSOR_SPAN_V = 0.010
DEVICE_CENTER_V = 0.0
DEVICE_SPAN_V = 0.050
SENSOR_POINTS = 100
DEVICE_POINTS = 60


def _default_param_overrides(**extra):
    """Merge test-default scan parameters with any caller overrides."""
    base = {
        "x_center": SENSOR_CENTER_V,
        "x_span": SENSOR_SPAN_V,
        "x_points": SENSOR_POINTS,
        "y_center": DEVICE_CENTER_V,
        "y_span": DEVICE_SPAN_V,
        "y_points": DEVICE_POINTS,
    }
    base.update(extra)
    return base


@pytest.mark.analysis
@pytest.mark.skipif(not _qarray_available(), reason="qarray/JAX not functional in this environment")
class TestSensorCompensationE2E:
    """End-to-end: qarray simulation -> inject ds_raw_all -> node analyse_data."""

    def test_node_analysis_single_pair(self, dot_model, analysis_runner):
        """Load node 01, inject one simulated sensor-vs-device scan, run analysis.

        Verifies the full pipeline: process_raw_dataset -> fit_shifted_lorentzian
        -> extract_sensor_compensation_coefficients, all through the node's
        registered run_action handlers.
        """
        v_sensor = sweep_voltages_mV(SENSOR_CENTER_V, SENSOR_SPAN_V, SENSOR_POINTS)
        v_device = sweep_voltages_mV(DEVICE_CENTER_V, DEVICE_SPAN_V, DEVICE_POINTS)

        ds_raw = simulate_sensor_device_scan(
            dot_model,
            v_sensor,
            v_device,
            sensor_gate_idx=6,
            device_gate_idx=0,
        )

        pair_key = "x_volts_vs_y_volts"
        node = analysis_runner(
            "01_sensor_gate_compensation",
            ds_raw_all={pair_key: ds_raw},
            param_overrides=_default_param_overrides(
                sensor_device_mapping={"x_volts": ["y_volts"]},
            ),
            artifacts_subdir="01_sensor_gate_compensation_e2e",
        )

        assert "fit_results" in node.results, "analyse_data did not produce fit_results"
        assert pair_key in node.results["fit_results"]

        fit = node.results["fit_results"][pair_key]
        alpha = fit["coefficient"]
        assert np.isfinite(alpha), f"alpha is not finite: {alpha}"
        assert fit["fit_params"]["success"], f"Fit did not converge: {fit}"

        # Verify the compensation matrix was updated: M[sensor, device] = α
        layer = node.machine.virtual_gate_sets["main_qpu"].layers[0]
        row = layer.source_gates.index("x_volts")
        col = layer.source_gates.index("y_volts")
        assert layer.matrix[row][col] == pytest.approx(alpha), (
            f"Expected matrix[{row}][{col}] = {alpha}, got {layer.matrix[row][col]}"
        )

    def test_node_analysis_two_pairs(self, dot_model, analysis_runner):
        """Run with two sensor-device pairs and verify both produce results."""
        v_sensor = sweep_voltages_mV(SENSOR_CENTER_V, SENSOR_SPAN_V, 80)
        v_device = sweep_voltages_mV(DEVICE_CENTER_V, DEVICE_SPAN_V, 50)

        ds_dot0 = simulate_sensor_device_scan(
            dot_model, v_sensor, v_device,
            sensor_gate_idx=6, device_gate_idx=0,
        )
        ds_dot1 = simulate_sensor_device_scan(
            dot_model, v_sensor, v_device,
            sensor_gate_idx=6, device_gate_idx=1,
        )

        node = analysis_runner(
            "01_sensor_gate_compensation",
            ds_raw_all={
                "x_volts_vs_y_volts_dot0": ds_dot0,
                "x_volts_vs_y_volts_dot1": ds_dot1,
            },
            param_overrides=_default_param_overrides(
                x_points=80,
                y_points=50,
                sensor_device_mapping={
                    "x_volts": ["y_volts_dot0", "y_volts_dot1"],
                },
            ),
            artifacts_subdir="01_sensor_gate_compensation_two_pairs",
        )

        assert len(node.results["fit_results"]) == 2
        layer = node.machine.virtual_gate_sets["main_qpu"].layers[0]
        for pair_key, fit in node.results["fit_results"].items():
            assert fit["fit_params"]["success"], f"Fit failed for {pair_key}: {fit}"
            assert np.isfinite(fit["coefficient"])
            sensor_gate, device_gate = pair_key.split("_vs_")
            row = layer.source_gates.index(sensor_gate)
            col = layer.source_gates.index(device_gate)
            assert layer.matrix[row][col] == pytest.approx(fit["coefficient"])

    def test_plot_fit_diagnostic(self, dot_model):
        """Generate a diagnostic plot of simulated data, Lorentzian fit, and residual."""
        v_sensor = sweep_voltages_mV(SENSOR_CENTER_V, SENSOR_SPAN_V, SENSOR_POINTS)
        v_device = sweep_voltages_mV(DEVICE_CENTER_V, DEVICE_SPAN_V, DEVICE_POINTS)

        ds_raw = simulate_sensor_device_scan(
            dot_model, v_sensor, v_device,
            sensor_gate_idx=6, device_gate_idx=0,
        )
        ds = process_raw_dataset(ds_raw)
        result = extract_sensor_compensation_coefficients(
            ds, sensor_gate_name="x_volts", device_gate_name="y_volts",
        )

        v_s = ds["amplitude"].coords["x_volts"].values
        v_d = ds["amplitude"].coords["y_volts"].values
        amplitude = ds["amplitude"].isel(sensors=0).values

        fp = result["fit_params"]
        model_signal = shifted_lorentzian_2d(
            v_s, v_d, fp["A"], fp["v0"], fp["alpha"], fp["gamma"], fp["offset"],
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        extent = [v_s[0], v_s[-1], v_d[0], v_d[-1]]

        axes[0].imshow(amplitude, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[0].set_title("Simulated data")
        axes[0].set_xlabel("Sensor gate (V)")
        axes[0].set_ylabel("Device gate (V)")

        axes[1].imshow(model_signal, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[1].set_title(f"Lorentzian fit (α={fp['alpha']:.4f})")
        axes[1].set_xlabel("Sensor gate (V)")

        residual = amplitude - model_signal
        axes[2].imshow(residual, extent=extent, origin="lower", aspect="auto", cmap="RdBu_r")
        axes[2].set_title("Residual")
        axes[2].set_xlabel("Sensor gate (V)")

        plt.tight_layout()

        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "01_sensor_compensation_qarray"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "fit_diagnostic.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "fit_diagnostic.png").exists()

    def test_plot_compensation_comparison(self, dot_model):
        """Compare original scan vs compensated (virtualized) scan."""
        v_sensor = sweep_voltages_mV(SENSOR_CENTER_V, SENSOR_SPAN_V, SENSOR_POINTS)
        v_device = sweep_voltages_mV(DEVICE_CENTER_V, DEVICE_SPAN_V, DEVICE_POINTS)

        ds_raw = simulate_sensor_device_scan(
            dot_model, v_sensor, v_device,
            sensor_gate_idx=6, device_gate_idx=0,
        )
        ds = process_raw_dataset(ds_raw)
        result = extract_sensor_compensation_coefficients(
            ds, sensor_gate_name="x_volts", device_gate_name="y_volts",
        )
        alpha = result["coefficient"]

        ds_raw_comp = simulate_sensor_device_scan(
            dot_model, v_sensor, v_device,
            sensor_gate_idx=6, device_gate_idx=0,
            compensation_alpha=alpha,
        )
        ds_comp = process_raw_dataset(ds_raw_comp)

        v_s = ds["amplitude"].coords["x_volts"].values
        v_d = ds["amplitude"].coords["y_volts"].values
        amp_orig = ds["amplitude"].isel(sensors=0).values
        amp_comp = ds_comp["amplitude"].isel(sensors=0).values
        extent = [v_s[0], v_s[-1], v_d[0], v_d[-1]]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(amp_orig, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[0].set_title(f"Original (α={alpha:.4f})")
        axes[0].set_xlabel("Sensor gate (V)")
        axes[0].set_ylabel("Device gate (V)")

        axes[1].imshow(amp_comp, extent=extent, origin="lower", aspect="auto", cmap="hot")
        axes[1].set_title("Compensated (virtual device gate)")
        axes[1].set_xlabel("Sensor gate (V)")
        axes[1].set_ylabel("Virtual device gate (V)")

        plt.tight_layout()

        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "01_sensor_compensation_qarray"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "compensation_comparison.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "compensation_comparison.png").exists()
