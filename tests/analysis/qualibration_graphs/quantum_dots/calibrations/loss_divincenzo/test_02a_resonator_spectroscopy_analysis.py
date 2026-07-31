"""Analysis test for 02a_resonator_spectroscopy.

Uses the node's ``generate_simulated_dataset`` helper to build synthetic I/Q
data, then runs the analysis pipeline via the shared ``analysis_runner`` fixture.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from matplotlib.figure import Figure
from qualang_tools.units import unit

from calibration_utils.resonator_spectroscopy import generate_simulated_dataset

NODE_NAME = "02a_resonator_spectroscopy"
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


def _expected_dip_shift_hz(frequency_span_in_mhz: float, seed: int = SIMULATION_SEED) -> float:
    """Match the first random dip position drawn by ``generate_simulated_dataset``."""
    u = unit(coerce_to_integer=True)
    span = frequency_span_in_mhz * u.MHz
    rng = np.random.default_rng(seed=seed)
    return float(rng.uniform(-span * 0.05, span * 0.05))


@pytest.mark.analysis
def test_02a_resonator_spectroscopy_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot actions and validate fit + figure outputs."""
    _ensure_werkzeug_serving_stub()
    _ensure_video_mode_parameters_stub()

    expected_dip_shift_hz = _expected_dip_shift_hz(FREQUENCY_SPAN_MHZ)

    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=generate_simulated_dataset,
        param_overrides={
            "num_shots": 8,
            "frequency_span_in_mhz": FREQUENCY_SPAN_MHZ,
            "frequency_step_in_mhz": 0.04,
            "sensor_names": [SENSOR_NAME],
        },
        analyse_qubits=[],
    )

    assert "IQ_abs" not in node.results["ds_raw"], "ds_raw should remain unprocessed after analysis."
    assert "IQ_abs" in node.results["ds_fit"], "Analysis preprocessing should add IQ_abs to ds_fit."
    assert "phase" in node.results["ds_fit"], "Analysis preprocessing should add phase to ds_fit."
    assert "full_freq" in node.results["ds_fit"].coords, "Processed data should include full_freq coordinate in ds_fit."

    assert "ds_fit" in node.results
    assert "fit_results" in node.results
    assert SENSOR_NAME in node.results["fit_results"]

    fit = node.results["fit_results"][SENSOR_NAME]
    assert fit["success"], f"Resonator fit should succeed, got: {fit}"

    fitted_shift = float(fit["frequency_shift"])
    assert abs(fitted_shift - expected_dip_shift_hz) < 0.8e6, (
        f"Expected dip near {expected_dip_shift_hz:.0f} Hz, got {fitted_shift:.0f} Hz"
    )

    fwhm = float(fit["fwhm"])
    assert np.isfinite(fwhm) and fwhm > 0.0, f"Expected positive finite FWHM, got {fwhm}"

    figures = node.results.get("figures")
    assert isinstance(figures, dict), "plot_data should store figures under node.results['figures']."
    assert {"phase", "amplitude"}.issubset(figures.keys()), "plot_data should create phase and amplitude figures."
    assert isinstance(figures["phase"], Figure)
    assert isinstance(figures["amplitude"], Figure)
    assert len(figures["phase"].axes) > 0
    assert len(figures["amplitude"].axes) > 0

    updated_if = float(node.machine.sensor_dots[SENSOR_NAME].readout_resonator.intermediate_frequency)
    assert np.isclose(
        updated_if, fit["resonator_frequency"], rtol=0.0, atol=1e-3
    ), f"Expected state IF update to {fit['resonator_frequency']}, got {updated_if}"
