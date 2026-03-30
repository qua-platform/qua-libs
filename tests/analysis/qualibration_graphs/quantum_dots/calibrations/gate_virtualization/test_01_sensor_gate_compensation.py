"""Analysis tests for node 01 — sensor gate compensation.

Uses the qarray ChargeSensedDotArray to simulate a 2D scan of the sensor
gate vs a device gate, then runs the node's full analysis pipeline
(analyse_data, plot_data, update_state) end-to-end.

The sensor Coulomb peak shifts linearly with the device gate voltage;
fitting a shifted Lorentzian recovers the cross-talk coefficient (alpha).

Gate names used here match the QuAM created by quam_factory:
  sensor gate : virtual_sensor_1
  device gates: virtual_dot_1, virtual_dot_2
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

try:
    from .simulation_helpers import simulate_sensor_device_scan, sweep_voltages_mV
    from .conftest import simulate_plunger_plunger_scan, simulate_sensor_sweep
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).parent))
    from simulation_helpers import simulate_sensor_device_scan, sweep_voltages_mV  # type: ignore[no-redef]
    from conftest import simulate_plunger_plunger_scan, simulate_sensor_sweep  # type: ignore[no-redef]

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

# Gate names that match the QuAM built by quam_factory.
SENSOR_GATE = "virtual_sensor_1"
DEVICE_GATE_1 = "virtual_dot_1"
DEVICE_GATE_2 = "virtual_dot_2"

SENSOR_CENTER_V = 0.015
SENSOR_SPAN_V = 0.0025
DEVICE_CENTER_V = 0.0
DEVICE_SPAN_V = 0.10
SENSOR_POINTS = 100
DEVICE_POINTS = 150


def _default_param_overrides(**extra):
    """Merge test-default scan parameters with any caller overrides."""
    base = {
        "sensor_gate_span": SENSOR_SPAN_V,
        "sensor_gate_points": SENSOR_POINTS,
        "device_gate_span": DEVICE_SPAN_V,
        "device_gate_points": DEVICE_POINTS,
        "fit_method": "bayesian_cp",
        "fit_kwargs": {"hazard": 1 / 30.0, "cp_threshold": 0.3},
    }
    base.update(extra)
    return base


def _qarray_available() -> bool:
    """Check whether qarray + JAX can actually run (catches version mismatches)."""
    try:
        from qarray import DotArray

        m = DotArray(Cdd=[[0.1]], Cgd=[[0.1]], algorithm="default", implementation="jax")
        m.ground_state_open(np.array([[0.0], [0.1]]))
        return True
    except Exception:
        return False


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

        pair_key = f"{SENSOR_GATE}_vs_{DEVICE_GATE_1}"
        node = analysis_runner(
            "01_sensor_gate_compensation",
            ds_raw_all={pair_key: ds_raw},
            param_overrides=_default_param_overrides(
                sensor_device_mapping={SENSOR_GATE: [DEVICE_GATE_1]},
            ),
            artifacts_subdir="01_sensor_compensation",
        )

        assert "fit_results" in node.results, "analyse_data did not produce fit_results"
        assert pair_key in node.results["fit_results"]

        fit = node.results["fit_results"][pair_key]
        assert fit["sensor_gate_name"] == SENSOR_GATE
        assert fit["device_gate_name"] == DEVICE_GATE_1
        alpha = fit["coefficient"]
        assert np.isfinite(alpha), f"alpha is not finite: {alpha}"
        assert fit["fit_params"]["success"], f"Fit did not converge: {fit}"

        # Verify update_state wrote alpha into the compensation matrix.
        # The matrix starts as identity, so M[sensor, device] should equal alpha.
        layer = node.machine.virtual_gate_sets["main_qpu"].layers[0]
        sensor_row = layer.source_gates.index(SENSOR_GATE)
        device_col = layer.source_gates.index(DEVICE_GATE_1)
        assert layer.matrix[sensor_row][device_col] == pytest.approx(alpha), (
            f"Expected matrix[{sensor_row}][{device_col}] = {alpha}, " f"got {layer.matrix[sensor_row][device_col]}"
        )

    def test_node_analysis_two_pairs(self, dot_model, analysis_runner):
        """Run with two sensor-device pairs and verify both produce results."""
        v_sensor = sweep_voltages_mV(SENSOR_CENTER_V, SENSOR_SPAN_V, 80)
        v_device = sweep_voltages_mV(DEVICE_CENTER_V, DEVICE_SPAN_V, 50)

        ds_dot1 = simulate_sensor_device_scan(
            dot_model,
            v_sensor,
            v_device,
            sensor_gate_idx=6,
            device_gate_idx=0,
        )
        ds_dot2 = simulate_sensor_device_scan(
            dot_model,
            v_sensor,
            v_device,
            sensor_gate_idx=6,
            device_gate_idx=1,
        )

        pair_key_1 = f"{SENSOR_GATE}_vs_{DEVICE_GATE_1}"
        pair_key_2 = f"{SENSOR_GATE}_vs_{DEVICE_GATE_2}"
        node = analysis_runner(
            "01_sensor_gate_compensation",
            ds_raw_all={
                pair_key_1: ds_dot1,
                pair_key_2: ds_dot2,
            },
            param_overrides=_default_param_overrides(
                sensor_gate_points=80,
                device_gate_points=50,
                sensor_device_mapping={SENSOR_GATE: [DEVICE_GATE_1, DEVICE_GATE_2]},
            ),
            artifacts_subdir="01_sensor_compensation",
        )

        assert len(node.results["fit_results"]) == 2
        layer = node.machine.virtual_gate_sets["main_qpu"].layers[0]
        sensor_row = layer.source_gates.index(SENSOR_GATE)
        for pair_key, fit in node.results["fit_results"].items():
            assert fit["fit_params"]["success"], f"Fit failed for {pair_key}: {fit}"
            assert np.isfinite(fit["coefficient"])
            assert fit["sensor_gate_name"] == SENSOR_GATE
            _, device_gate = pair_key.split("_vs_")
            assert fit["device_gate_name"] == device_gate
            device_col = layer.source_gates.index(device_gate)
            assert layer.matrix[sensor_row][device_col] == pytest.approx(fit["coefficient"])

    def test_plot_fit_diagnostic(self, dot_model, analysis_runner):
        """node plot_data produces a 3-panel diagnostic figure (data / fit / residual)."""
        v_sensor = sweep_voltages_mV(SENSOR_CENTER_V, SENSOR_SPAN_V, SENSOR_POINTS)
        v_device = sweep_voltages_mV(DEVICE_CENTER_V, DEVICE_SPAN_V, DEVICE_POINTS)

        ds_raw = simulate_sensor_device_scan(
            dot_model,
            v_sensor,
            v_device,
            sensor_gate_idx=6,
            device_gate_idx=0,
        )
        pair_key = f"{SENSOR_GATE}_vs_{DEVICE_GATE_1}"
        node = analysis_runner(
            "01_sensor_gate_compensation",
            ds_raw_all={pair_key: ds_raw},
            param_overrides=_default_param_overrides(
                sensor_device_mapping={SENSOR_GATE: [DEVICE_GATE_1]},
            ),
            artifacts_subdir="01_sensor_compensation",
        )

        figures = node.results.get("figures", {})
        assert len(figures) > 0, "plot_data did not produce a figure"

    def test_plot_compensation_comparison(self, dot_model, analysis_runner):
        """Alpha from node's analyse_data correctly removes cross-talk in a re-simulated scan."""
        v_sensor = sweep_voltages_mV(SENSOR_CENTER_V, SENSOR_SPAN_V, SENSOR_POINTS)
        v_device = sweep_voltages_mV(DEVICE_CENTER_V, DEVICE_SPAN_V, DEVICE_POINTS)

        ds_raw = simulate_sensor_device_scan(
            dot_model,
            v_sensor,
            v_device,
            sensor_gate_idx=6,
            device_gate_idx=0,
        )
        pair_key = f"{SENSOR_GATE}_vs_{DEVICE_GATE_1}"
        node = analysis_runner(
            "01_sensor_gate_compensation",
            ds_raw_all={pair_key: ds_raw},
            param_overrides=_default_param_overrides(
                sensor_device_mapping={SENSOR_GATE: [DEVICE_GATE_1]},
            ),
            artifacts_subdir="01_sensor_compensation",
        )

        alpha = node.results["fit_results"][pair_key]["coefficient"]

        # Re-simulate with compensation applied and build a side-by-side comparison plot.
        ds_raw_comp = simulate_sensor_device_scan(
            dot_model,
            v_sensor,
            v_device,
            sensor_gate_idx=6,
            device_gate_idx=0,
            compensation_alpha=alpha,
        )
        ds = process_raw_dataset(ds_raw)
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

        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "01_sensor_compensation"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "compensation_comparison.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "compensation_comparison.png").exists()

    def test_plot_plunger_sweep_before_after(self, dot_model, analysis_runner):
        """Show effect of sensor compensation on a dot_1-vs-dot_2 charge stability scan."""
        # Run node analysis for both device gates to get alphas
        v_sensor = sweep_voltages_mV(SENSOR_CENTER_V, SENSOR_SPAN_V, SENSOR_POINTS)
        v_device = sweep_voltages_mV(DEVICE_CENTER_V, DEVICE_SPAN_V, DEVICE_POINTS)

        ds_raw_all = {}
        for gate_name, gate_idx in [(DEVICE_GATE_1, 0), (DEVICE_GATE_2, 1)]:
            ds_raw_all[f"{SENSOR_GATE}_vs_{gate_name}"] = simulate_sensor_device_scan(
                dot_model,
                v_sensor,
                v_device,
                sensor_gate_idx=6,
                device_gate_idx=gate_idx,
            )

        node = analysis_runner(
            "01_sensor_gate_compensation",
            ds_raw_all=ds_raw_all,
            param_overrides=_default_param_overrides(
                sensor_device_mapping={SENSOR_GATE: [DEVICE_GATE_1, DEVICE_GATE_2]},
            ),
            artifacts_subdir="01_sensor_compensation",
        )

        alphas = {
            0: node.results["fit_results"][f"{SENSOR_GATE}_vs_{DEVICE_GATE_1}"]["coefficient"],
            1: node.results["fit_results"][f"{SENSOR_GATE}_vs_{DEVICE_GATE_2}"]["coefficient"],
        }
        for pair_key, fit in node.results["fit_results"].items():
            fp = fit["fit_params"]
            n_cp = fp.get("n_changepoints", 0)
            alpha_std = fp.get("alpha_std", float("nan"))
            print(f"  {pair_key}: α={fit['coefficient']:.6f} ± {alpha_std:.6f}  ({n_cp} CPs)")

        # Find sensor operating point
        sensor_sweep_mV = np.linspace(
            (SENSOR_CENTER_V - SENSOR_SPAN_V) * 1e3,
            (SENSOR_CENTER_V + SENSOR_SPAN_V) * 1e3,
            300,
        )
        ds_sensor = simulate_sensor_sweep(dot_model, sensor_sweep_mV)
        signal = np.hypot(ds_sensor["I"].values[0], ds_sensor["Q"].values[0])
        sensor_op_mV = float(sensor_sweep_mV[np.argmax(signal)])

        plunger_span_V = 0.050
        plunger_pts = 200
        v_p = sweep_voltages_mV(0.0, plunger_span_V, plunger_pts)

        data_uncomp = simulate_plunger_plunger_scan(
            dot_model,
            v_p,
            v_p,
            plunger_x_gate_idx=0,
            plunger_y_gate_idx=1,
            sensor_operating_point=sensor_op_mV,
        )
        data_comp = simulate_plunger_plunger_scan(
            dot_model,
            v_p,
            v_p,
            plunger_x_gate_idx=0,
            plunger_y_gate_idx=1,
            sensor_operating_point=sensor_op_mV,
            sensor_compensation=alphas,
        )

        amp_uncomp = np.hypot(
            data_uncomp["I"].values[0],
            data_uncomp["Q"].values[0],
        )
        amp_comp = np.hypot(
            data_comp["I"].values[0],
            data_comp["Q"].values[0],
        )
        extent = [v_p[0], v_p[-1], v_p[0], v_p[-1]]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, data, title in [
            (axes[0], amp_uncomp, "Physical (no sensor comp)"),
            (axes[1], amp_comp, f"Sensor-compensated (α₁={alphas[0]:.4f}, α₂={alphas[1]:.4f})"),
        ]:
            ax.imshow(data, extent=extent, origin="lower", aspect="auto", cmap="hot")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("dot_1 (mV)")
            ax.set_ylabel("dot_2 (mV)")
        fig.suptitle("Plunger Sweep: Effect of Sensor Compensation", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "01_sensor_compensation"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "plunger_sweep_comparison.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "plunger_sweep_comparison.png").exists()


@pytest.mark.analysis
@pytest.mark.skipif(not _qarray_available(), reason="qarray/JAX not functional in this environment")
class TestSensorDotCouplingEffect:
    """Demonstrate how sensor-gate → device-dot capacitive coupling tilts
    charge transitions and degrades the sensor compensation fit.

    When the sensor plunger has non-zero Cgd to device dots, sweeping the
    sensor gate also shifts the dot chemical potentials.  Charge transitions
    in the (sensor, device) 2D scan are no longer horizontal — they acquire
    a slope proportional to the coupling ratio Cgd_sensor→dot / Cgd_dot→dot.
    The BCP analysis assumes piecewise-constant peak positions (horizontal
    transitions) and will mis-estimate alpha when they are tilted.
    """

    COUPLING_STRENGTHS = [0.0, 0.01, 0.03, 0.06]

    @staticmethod
    def _make_model(sensor_dot_coupling: float):
        """Build a qarray model with controllable sensor→dot coupling.

        Cgd[dot_0, gate_sensor] = sensor_dot_coupling
        Cgd[dot_1, gate_sensor] = sensor_dot_coupling * 0.5

        For reference, the main plunger self-capacitances are 0.13 (dot_0)
        and 0.11 (dot_1), so coupling = 0.06 is ~46% of the self-coupling
        — a significant parasitic effect.
        """
        from validation_utils.charge_stability.default import init_dot_model as _idm

        Cgd = [
            [0.13, 0.00, 0.00, 0.00, 0.00, 0.00, sensor_dot_coupling],
            [0.00, 0.11, 0.00, 0.00, 0.00, 0.00, sensor_dot_coupling * 0.5],
            [0.00, 0.00, 0.09, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.13, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.13, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00],
        ]
        return _idm(
            Cgd=Cgd,
            Cds=[[0.003, 0.0015, 0.002, 0.002, 0.002, 0.002]],
            Cgs=[[0.0015, 0.001, 0.000, 0.000, 0.000, 0.000, 0.100]],
        )

    def test_coupling_effect_on_sensor_compensation(self, analysis_runner):
        """Sweep sensor-dot coupling and show how it degrades alpha estimation.

        For each coupling strength:
        1. Simulate a sensor-vs-dot_1 scan.
        2. Run the full node 01 analysis pipeline.
        3. Record the fitted alpha, whether the fit succeeded, and the
           scan image (showing tilted vs horizontal transitions).

        Generates a multi-panel figure comparing the raw scans and a
        summary table of fitted alphas vs coupling strength.
        """
        v_sensor = sweep_voltages_mV(SENSOR_CENTER_V, SENSOR_SPAN_V, SENSOR_POINTS)
        v_device = sweep_voltages_mV(DEVICE_CENTER_V, DEVICE_SPAN_V, DEVICE_POINTS)
        pair_key = f"{SENSOR_GATE}_vs_{DEVICE_GATE_1}"

        results_by_coupling = {}

        for coupling in self.COUPLING_STRENGTHS:
            model = self._make_model(coupling)

            ds_raw = simulate_sensor_device_scan(
                model,
                v_sensor,
                v_device,
                sensor_gate_idx=6,
                device_gate_idx=0,
            )

            node = analysis_runner(
                "01_sensor_gate_compensation",
                ds_raw_all={pair_key: ds_raw},
                param_overrides=_default_param_overrides(
                    sensor_device_mapping={SENSOR_GATE: [DEVICE_GATE_1]},
                ),
                artifacts_subdir=f"01_sensor_coupling/coupling_{coupling:.3f}",
            )

            fit = node.results.get("fit_results", {}).get(pair_key, {})
            fp = fit.get("fit_params", {})
            ds_proc = process_raw_dataset(ds_raw)
            amp = ds_proc["amplitude"].isel(sensors=0).values

            results_by_coupling[coupling] = {
                "alpha": fit.get("coefficient", float("nan")),
                "success": fp.get("success", False),
                "n_changepoints": fp.get("n_changepoints", 0),
                "alpha_std": fp.get("alpha_std", float("nan")),
                "amplitude_2d": amp,
                "v_sensor": ds_proc["amplitude"].coords["x_volts"].values,
                "v_device": ds_proc["amplitude"].coords["y_volts"].values,
            }

        # Build comparison figure
        n_cols = len(self.COUPLING_STRENGTHS)
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))

        for col, coupling in enumerate(self.COUPLING_STRENGTHS):
            r = results_by_coupling[coupling]
            extent = [r["v_sensor"][0], r["v_sensor"][-1],
                      r["v_device"][0], r["v_device"][-1]]

            ax_scan = axes[0, col]
            ax_scan.imshow(r["amplitude_2d"], extent=extent,
                           origin="lower", aspect="auto", cmap="hot")
            ax_scan.set_title(
                f"Cgd(sensor→dot) = {coupling:.3f}\n"
                f"α = {r['alpha']:.6f}  (CPs: {r['n_changepoints']})",
                fontsize=9,
            )
            ax_scan.set_xlabel("Sensor gate (V)")
            ax_scan.set_ylabel("Device gate (V)")

            # Also run with compensation to show residual
            model = self._make_model(coupling)
            ds_comp = simulate_sensor_device_scan(
                model, v_sensor, v_device,
                sensor_gate_idx=6, device_gate_idx=0,
                compensation_alpha=r["alpha"],
            )
            ds_comp_proc = process_raw_dataset(ds_comp)
            amp_comp = ds_comp_proc["amplitude"].isel(sensors=0).values

            ax_comp = axes[1, col]
            ax_comp.imshow(amp_comp, extent=extent,
                           origin="lower", aspect="auto", cmap="hot")
            ax_comp.set_title(
                f"After compensation (α = {r['alpha']:.6f})",
                fontsize=9,
            )
            ax_comp.set_xlabel("Sensor gate (V)")
            ax_comp.set_ylabel("Virtual device gate (V)")

        fig.suptitle(
            "Effect of Sensor-Dot Capacitive Coupling on Sensor Compensation\n"
            "Top: raw scan  |  Bottom: after applying fitted α",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "01_sensor_coupling"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "coupling_effect_comparison.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "coupling_effect_comparison.png").exists()

        # Print summary table
        print("\n=== Sensor-Dot Coupling Effect Summary ===")
        print(f"{'Cgd':>8s} {'alpha':>12s} {'alpha_std':>12s} {'CPs':>5s} {'success':>8s}")
        for coupling in self.COUPLING_STRENGTHS:
            r = results_by_coupling[coupling]
            print(
                f"{coupling:8.3f} {r['alpha']:12.6f} {r['alpha_std']:12.6f} "
                f"{r['n_changepoints']:5d} {str(r['success']):>8s}"
            )

        # The zero-coupling case should succeed; we don't assert failure
        # for coupled cases — this test is diagnostic, showing the
        # degradation trend.
        r0 = results_by_coupling[0.0]
        assert r0["success"], "Zero-coupling baseline should succeed"
        assert np.isfinite(r0["alpha"]), "Zero-coupling alpha should be finite"
