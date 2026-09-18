"""Analysis test for 02b_resonator_spectroscopy_vs_power.

Uses the node's ``generate_simulated_dataset`` helper to build synthetic 2D I/Q
data (frequency × readout power), then runs the analysis pipeline via the shared
``analysis_runner`` fixture.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from matplotlib.figure import Figure
from qualang_tools.units import unit

from calibration_utils.resonator_spectroscopy_vs_power import generate_simulated_dataset

NODE_NAME = "02b_resonator_spectroscopy_vs_power"
SENSOR_NAME = "virtual_sensor_1"
FREQUENCY_SPAN_MHZ = 12.0
SIMULATION_SEED = 42


def _ensure_werkzeug_serving_stub() -> None:
    """Provide a minimal werkzeug.serving stub for optional video-mode imports."""
    try:
        from werkzeug.serving import make_server  # noqa: F401
    except ModuleNotFoundError:
        werkzeug_mod = types.ModuleType("werkzeug")
        serving_mod = types.ModuleType("werkzeug.serving")

        def _make_server(*args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("werkzeug is not available in this test environment.")

        serving_mod.make_server = _make_server
        werkzeug_mod.serving = serving_mod
        sys.modules["werkzeug"] = werkzeug_mod
        sys.modules["werkzeug.serving"] = serving_mod


def _ensure_video_mode_parameters_stub() -> None:
    """Stub optional video-mode parameter mixin when UI deps are missing."""
    try:
        from calibration_utils.run_video_mode.video_mode_specific_parameters import (  # noqa: F401
            VideoModeCommonParameters,
        )
    except ModuleNotFoundError:
        from qualibrate.core.parameters import RunnableParameters

        package_mod = types.ModuleType("calibration_utils.run_video_mode")
        params_mod = types.ModuleType("calibration_utils.run_video_mode.video_mode_specific_parameters")

        class VideoModeCommonParameters(RunnableParameters):
            """Minimal stub used only for analysis tests."""

        params_mod.VideoModeCommonParameters = VideoModeCommonParameters
        package_mod.video_mode_specific_parameters = params_mod
        sys.modules["calibration_utils.run_video_mode"] = package_mod
        sys.modules["calibration_utils.run_video_mode.video_mode_specific_parameters"] = params_mod


def _expected_dip_center_hz(frequency_span_in_mhz: float, seed: int = SIMULATION_SEED) -> float:
    """Match ``dip_center`` drawn by ``generate_simulated_dataset`` (after ``kappa_base``)."""
    u = unit(coerce_to_integer=True)
    span = frequency_span_in_mhz * u.MHz
    rng = np.random.default_rng(seed=seed)
    rng.uniform(0.5e6, 1.5e6)  # kappa_base — consumed before dip_center
    return float(rng.uniform(-span * 0.05, span * 0.05))


@pytest.mark.analysis
def test_02b_resonator_spectroscopy_vs_power_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot/update actions and validate fit + figure outputs."""
    _ensure_werkzeug_serving_stub()
    _ensure_video_mode_parameters_stub()

    min_power_dbm = -50
    max_power_dbm = -25

    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=generate_simulated_dataset,
        param_overrides={
            "num_shots": 8,
            "frequency_span_in_mhz": FREQUENCY_SPAN_MHZ,
            "frequency_step_in_mhz": 0.04,
            "sensor_names": [SENSOR_NAME],
            "min_power_dbm": min_power_dbm,
            "max_power_dbm": max_power_dbm,
            "num_power_points": 40,
        },
        analyse_qubits=[],
    )

    ds_raw = node.results["ds_raw"]
    assert "IQ_abs" not in ds_raw, "ds_raw should remain unprocessed after analysis."
    assert set(ds_raw.dims) >= {"sensor", "frequency_detuning", "power"}

    ds_fit = node.results["ds_fit"]
    assert "IQ_abs" in ds_fit
    assert "IQ_abs_norm" in ds_fit
    assert "phase" in ds_fit
    assert "full_freq" in ds_fit.coords
    assert "optimal_power" in ds_fit.coords
    assert "frequency_shift" in ds_fit.coords

    assert SENSOR_NAME in node.results["fit_results"]
    fit = node.results["fit_results"][SENSOR_NAME]
    assert fit["success"], f"Resonator vs power fit should succeed, got: {fit}"

    optimal_power = float(fit["optimal_power"])
    assert min_power_dbm <= optimal_power <= max_power_dbm, f"optimal_power {optimal_power:.2f} dBm outside sweep range"

    fitted_shift = float(fit["frequency_shift"])
    expected_dip_hz = _expected_dip_center_hz(FREQUENCY_SPAN_MHZ)
    assert (
        abs(fitted_shift - expected_dip_hz) < 1.0e6
    ), f"Expected frequency shift near {expected_dip_hz:.0f} Hz, got {fitted_shift:.0f} Hz"

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert "amplitude" in figures
    assert isinstance(figures["amplitude"], Figure)
    assert len(figures["amplitude"].axes) > 0

    resonator = node.machine.sensor_dots[SENSOR_NAME].readout_resonator
    assert np.isclose(
        float(resonator.intermediate_frequency),
        fit["resonator_frequency"],
        rtol=0.0,
        atol=1e-3,
    ), "State IF should match fitted resonator_frequency after update_state."

    readout_amp = float(resonator.operations["readout"].amplitude)
    expected_amp = unit(coerce_to_integer=True).dBm2volts(optimal_power, Z=50)
    assert np.isclose(
        readout_amp, expected_amp, rtol=0.0, atol=1e-6
    ), f"Readout amplitude should match optimal_power {optimal_power:.2f} dBm"
