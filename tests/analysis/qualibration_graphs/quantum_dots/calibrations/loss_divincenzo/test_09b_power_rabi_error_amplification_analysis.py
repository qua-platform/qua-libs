"""Analysis test for 09b_power_rabi_error_amplification.

Uses ``generate_simulated_dataset`` from calibration_utils and the shared
``analysis_runner`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils.power_rabi_error_amplification import generate_simulated_dataset

from .analysis_test_stubs import ensure_optional_analysis_import_stubs

NODE_NAME = "09b_power_rabi_error_amplification"
QUBIT_NAME = "q1"


@pytest.mark.analysis
def test_09b_power_rabi_error_amplification_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot/update on synthetic error-amplified power-Rabi data."""
    ensure_optional_analysis_import_stubs()

    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=generate_simulated_dataset,
        param_overrides={
            "num_shots": 8,
            "qubits": [QUBIT_NAME],
        },
    )

    ds_raw = node.results["ds_raw"]
    assert f"p_{QUBIT_NAME}" in ds_raw.data_vars
    assert set(ds_raw.dims) >= {"n_pulses", "amp_prefactor"}
    assert not any(name.startswith("E_") for name in ds_raw.data_vars)

    fit = node.results["fit_results"][QUBIT_NAME]
    assert fit["success"], f"Error-amplification fit should succeed, got: {fit}"

    opt_amp = float(fit["opt_amp"])
    assert node.parameters.min_amp_factor < opt_amp < node.parameters.max_amp_factor

    omega = float(fit["rabi_frequency"])
    gamma = float(fit["decay_rate"])
    assert np.isfinite(omega) and omega > 0
    assert np.isfinite(gamma)

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert {"heatmap", "resonance"}.issubset(figures.keys())
    for key in ("heatmap", "resonance"):
        assert isinstance(figures[key], Figure)
        assert len(figures[key].axes) > 0

    x180 = node.machine.qubits[QUBIT_NAME].macros["x180"]
    assert float(x180.pi_pulse.amplitude) > 0
