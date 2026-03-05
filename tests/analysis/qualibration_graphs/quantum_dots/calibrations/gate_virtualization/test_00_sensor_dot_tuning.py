"""Analysis tests for node 00 — sensor dot tuning.

Uses qarray to simulate a 1D sensor gate sweep across a Coulomb peak,
then fits a Lorentzian and verifies the extracted operating point sits
at the inflection point x0 ± γ/(2√3).
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from .conftest import sweep_voltages_mV

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


_analysis_mod = _load_module("_sd_analysis", _GATE_VIRT_UTILS / "sensor_dot_analysis.py")
fit_lorentzian = _analysis_mod.fit_lorentzian
lorentzian = _analysis_mod.lorentzian
optimal_operating_point = _analysis_mod.optimal_operating_point


# ── Test constants ───────────────────────────────────────────────────────────

SENSOR_CENTER_MV = 5.0
SENSOR_SPAN_MV = 6.0
SENSOR_POINTS = 300


def _qarray_available() -> bool:
    try:
        from qarray import DotArray

        m = DotArray(Cdd=[[0.1]], Cgd=[[0.1]], algorithm="default", implementation="jax")
        m.ground_state_open(np.array([[0.0], [0.1]]))
        return True
    except Exception:
        return False


def _simulate_sensor_sweep(model, v_sensor_mV: np.ndarray, sensor_gate_idx: int = 6) -> xr.Dataset:
    """1D sweep of the sensor gate — all other gates at zero."""
    n_gates = sensor_gate_idx + 1
    voltage_array = np.zeros((len(v_sensor_mV), n_gates))
    voltage_array[:, sensor_gate_idx] = v_sensor_mV

    z, _ = model.charge_sensor_open(-voltage_array)
    z = z.squeeze()

    v_V = v_sensor_mV * 1e-3
    I_data = z[np.newaxis, :]
    Q_data = np.zeros_like(I_data)

    return xr.Dataset(
        {
            "I": xr.DataArray(
                I_data,
                dims=["sensors", "x_volts"],
                coords={"sensors": ["sensor_1"], "x_volts": v_V},
            ),
            "Q": xr.DataArray(
                Q_data,
                dims=["sensors", "x_volts"],
                coords={"sensors": ["sensor_1"], "x_volts": v_V},
            ),
        }
    )


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.analysis
@pytest.mark.skipif(not _qarray_available(), reason="qarray/JAX not functional")
class TestSensorDotTuning:

    def test_lorentzian_fit_recovers_peak(self, dot_model):
        """Fit a Lorentzian to a simulated sensor sweep and check x0 recovery."""
        v_mV = np.linspace(
            SENSOR_CENTER_MV - SENSOR_SPAN_MV / 2,
            SENSOR_CENTER_MV + SENSOR_SPAN_MV / 2,
            SENSOR_POINTS,
        )
        ds = _simulate_sensor_sweep(dot_model, v_mV)

        signal = np.sqrt(ds["I"].values[0] ** 2 + ds["Q"].values[0] ** 2)
        v_V = ds.coords["x_volts"].values

        result = fit_lorentzian(v_V, signal, side="right")

        assert result.x0 == pytest.approx(SENSOR_CENTER_MV * 1e-3, abs=2e-3)
        assert result.gamma > 0
        assert result.optimal_voltage > result.x0

    def test_operating_point_formula(self):
        """Verify the operating point sits at x0 ± γ/(2√3)."""
        x0, gamma = 0.005, 0.003
        right = optimal_operating_point(x0, gamma, "right")
        left = optimal_operating_point(x0, gamma, "left")
        delta = gamma / (2 * np.sqrt(3))

        assert right == pytest.approx(x0 + delta)
        assert left == pytest.approx(x0 - delta)

    def test_plot_sensor_sweep(self, dot_model):
        """Generate a diagnostic plot: data, fit, and marked operating point."""
        v_mV = np.linspace(
            SENSOR_CENTER_MV - SENSOR_SPAN_MV / 2,
            SENSOR_CENTER_MV + SENSOR_SPAN_MV / 2,
            SENSOR_POINTS,
        )
        ds = _simulate_sensor_sweep(dot_model, v_mV)

        signal = np.sqrt(ds["I"].values[0] ** 2 + ds["Q"].values[0] ** 2)
        v_V = ds.coords["x_volts"].values

        result = fit_lorentzian(v_V, signal, side="right")

        v_fit = np.linspace(v_V[0], v_V[-1], 500)
        y_fit = lorentzian(v_fit, result.x0, result.gamma, result.amplitude, result.offset)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(v_V * 1e3, signal, "k.", markersize=3, label="Simulated data")
        ax.plot(v_fit * 1e3, y_fit, "r-", linewidth=1.5, label="Lorentzian fit")
        ax.axvline(result.x0 * 1e3, color="blue", linestyle="--", alpha=0.6, label=f"x0 = {result.x0 * 1e3:.2f} mV")
        ax.axvline(
            result.optimal_voltage * 1e3,
            color="green",
            linewidth=2,
            label=f"Op. pt = {result.optimal_voltage * 1e3:.2f} mV",
        )
        ax.set_xlabel("Sensor gate voltage (mV)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.set_title("Sensor dot tuning — Lorentzian fit")
        ax.legend(fontsize=8)
        plt.tight_layout()

        artifacts_dir = Path(__file__).resolve().parents[4] / "artifacts" / "00_sensor_dot_tuning"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(artifacts_dir / "sensor_sweep.png", dpi=150)
        plt.close(fig)
        assert (artifacts_dir / "sensor_sweep.png").exists()
