"""Analysis test for 09a_power_rabi.

Uses ``generate_simulated_dataset`` from calibration_utils and the shared
``analysis_runner`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils.power_rabi import generate_simulated_dataset

from .analysis_test_stubs import ensure_optional_analysis_import_stubs

NODE_NAME = "09a_power_rabi"
QUBIT_NAME = "q1"


@pytest.mark.analysis
def test_09a_power_rabi_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot/update on synthetic power-Rabi data."""
    ensure_optional_analysis_import_stubs()

    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=generate_simulated_dataset,
        param_overrides={
            "num_shots": 8,
            "min_amp_factor": 0.001,
            "max_amp_factor": 1.99,
            "amp_factor_step": 0.01,
            "qubits": [QUBIT_NAME],
        },
    )

    ds_raw = node.results["ds_raw"]
    assert f"p_{QUBIT_NAME}" in ds_raw.data_vars
    assert not any(name.startswith("E_") for name in ds_raw.data_vars)

    assert "ds_fit" in node.results
    assert "fit_results" in node.results
    fit = node.results["fit_results"][QUBIT_NAME]
    assert fit["success"], f"Power-Rabi fit should succeed, got: {fit}"

    opt_amp = float(fit["opt_amp"])
    assert 0.05 < opt_amp < 1.95, f"Expected opt_amp in [0.05, 1.95], got {opt_amp:.4f}"

    omega = float(fit["rabi_frequency"])
    gamma = float(fit["decay_rate"])
    assert np.isfinite(omega) and omega > 0
    assert np.isfinite(gamma)

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert {"rabi", "fft"}.issubset(figures.keys())
    for key in ("rabi", "fft"):
        assert isinstance(figures[key], Figure)
        assert len(figures[key].axes) > 0

    x180 = node.machine.qubits[QUBIT_NAME].macros["x180"]
    assert float(x180.pi_pulse.amplitude) > 0
