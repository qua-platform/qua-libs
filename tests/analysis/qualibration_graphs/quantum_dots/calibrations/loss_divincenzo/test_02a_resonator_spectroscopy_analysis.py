"""Analysis test for 02a_resonator_spectroscopy.

Builds a synthetic resonator-response dataset (I/Q vs detuning) with a
single Lorentzian dip, then runs the node analysis pipeline via the
shared ``analysis_runner`` fixture.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure

NODE_NAME = "02a_resonator_spectroscopy"
SENSOR_NAME = "virtual_sensor_1"
DETUNING_CENTER_HZ = 1.2e6


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
        from qualibrate.parameters import RunnableParameters

        package_mod = types.ModuleType("calibration_utils.run_video_mode")
        params_mod = types.ModuleType("calibration_utils.run_video_mode.video_mode_specific_parameters")

        class VideoModeCommonParameters(RunnableParameters):
            """Minimal stub used only for analysis tests."""

        params_mod.VideoModeCommonParameters = VideoModeCommonParameters
        package_mod.video_mode_specific_parameters = params_mod
        sys.modules["calibration_utils.run_video_mode"] = package_mod
        sys.modules["calibration_utils.run_video_mode.video_mode_specific_parameters"] = params_mod


def _build_resonator_ds_raw() -> xr.Dataset:
    """Return synthetic raw data in the format expected by the node."""
    detuning = np.linspace(-6e6, 6e6, 301, dtype=float)

    width_hz = 0.7e6
    baseline = 1.0
    depth = 0.35

    dip = depth / (1.0 + ((detuning - DETUNING_CENTER_HZ) / width_hz) ** 2)

    # Mild baseline tilt and quadrature component for realistic IQ traces.
    i_trace = baseline - dip + 0.01 * (detuning / np.max(np.abs(detuning)))
    q_trace = 0.02 * np.sin(2.0 * np.pi * (detuning - detuning.min()) / (detuning.ptp()))

    return xr.Dataset(
        data_vars={
            "I": (("sensors", "detuning"), i_trace[np.newaxis, :]),
            "Q": (("sensors", "detuning"), q_trace[np.newaxis, :]),
        },
        coords={
            "sensors": [SENSOR_NAME],
            "detuning": xr.DataArray(
                detuning,
                dims="detuning",
                attrs={"long_name": "readout frequency", "units": "Hz"},
            ),
        },
    )


@pytest.mark.analysis
def test_02a_resonator_spectroscopy_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot actions and validate fit + figure outputs."""
    _ensure_werkzeug_serving_stub()
    _ensure_video_mode_parameters_stub()
    ds_raw = _build_resonator_ds_raw()

    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "num_shots": 8,
            "frequency_span_in_mhz": 12,
            "frequency_step_in_mhz": 0.04,
            "sensor_names": [SENSOR_NAME],
        },
    )

    assert "IQ_abs" in node.results["ds_raw"], "Analysis preprocessing should add IQ_abs."
    assert "phase" in node.results["ds_raw"], "Analysis preprocessing should add phase."
    assert "full_freq" in node.results["ds_raw"].coords, "Processed data should include full_freq coordinate."

    assert "ds_fit" in node.results
    assert "fit_results" in node.results
    assert SENSOR_NAME in node.results["fit_results"]

    fit = node.results["fit_results"][SENSOR_NAME]
    assert fit["success"], f"Resonator fit should succeed, got: {fit}"

    fitted_detuning = float(node.results["ds_fit"].sel(sensors=SENSOR_NAME).position.values)
    assert (
        abs(fitted_detuning - DETUNING_CENTER_HZ) < 0.8e6
    ), f"Expected dip near {DETUNING_CENTER_HZ:.0f} Hz, got {fitted_detuning:.0f} Hz"

    fwhm = float(fit["fwhm"])
    assert np.isfinite(fwhm) and fwhm > 0.0, f"Expected positive finite FWHM, got {fwhm}"

    figures = node.results.get("figures")
    assert isinstance(figures, dict), "plot_data should store figures under node.results['figures']."
    assert {"phase", "amplitude"}.issubset(figures.keys()), "plot_data should create phase and amplitude figures."
    assert isinstance(figures["phase"], Figure)
    assert isinstance(figures["amplitude"], Figure)
    assert len(figures["phase"].axes) > 0
    assert len(figures["amplitude"].axes) > 0

    # update_state should set the sensor IF to the fitted resonance frequency.
    updated_if = float(node.machine.sensor_dots[SENSOR_NAME].readout_resonator.intermediate_frequency)
    assert np.isclose(
        updated_if, fit["frequency"], rtol=0.0, atol=1e-3
    ), f"Expected state IF update to {fit['frequency']}, got {updated_if}"
