"""Analysis test for ``03a_sensor_gate_sweep_opx``."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils.sensor_dot.simulated_data_generator import generate_simulated_dataset

NODE_NAME = "03a_sensor_gate_sweep_opx"
SENSOR_NAME = "virtual_sensor_1"


@pytest.mark.analysis
def test_03a_sensor_gate_sweep_opx_analysis_and_plot(analysis_runner):
    """Fit Lorentzian peaks for all sensors and generate analysis artifacts."""
    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=generate_simulated_dataset,
        param_overrides={
            "num_shots": 10,
            "offset_min": -0.2,
            "offset_max": 0.2,
            "offset_step": 0.01,
            "peak_fit_side": "left",
            "use_simulated_data": False,
            "simulate": False,
        },
    )

    assert "ds_fit" in node.results
    assert "fit_results" in node.results

    sensor_names = [s.name for s in node.namespace["sensors"]]
    for sensor_name in sensor_names:
        fit = node.results["fit_results"][sensor_name]
        assert fit["success"], f"Expected successful fit for {sensor_name}, got: {fit}"
        assert np.isfinite(fit["optimal_bias"])
        assert np.isfinite(fit["lorentzian_gamma"]) and fit["lorentzian_gamma"] > 0

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert {"phase", "amplitude_gradient"}.issubset(figures.keys())
    assert isinstance(figures["phase"], Figure)
    assert isinstance(figures["amplitude_gradient"], Figure)

    assert node.outcomes[SENSOR_NAME] == "successful"
