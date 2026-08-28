"""Analysis test for 09a_power_rabi.

Uses ``generate_simulated_dataset`` from calibration_utils and the shared
``analysis_runner`` fixture. ``q2`` is given structureless data so its fit
fails while ``q1`` succeeds, validating mixed outcomes and selective state updates.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils.power_rabi import generate_simulated_dataset
from analysis_helpers import snapshot_qubit_calibration, with_unfittable_qubit

NODE_NAME = "09a_power_rabi"
QUBIT_NAMES = ["q1", "q2"]
SUCCESS_QUBIT = "q1"
FAILING_QUBIT = "q2"


@pytest.mark.analysis
def test_09a_power_rabi_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot/update on synthetic power-Rabi data with one failing qubit."""
    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=with_unfittable_qubit(generate_simulated_dataset, FAILING_QUBIT),
        param_overrides={
            "num_shots": 8,
            "min_amp_factor": 0.001,
            "max_amp_factor": 1.99,
            "amp_factor_step": 0.01,
            "qubits": QUBIT_NAMES,
        },
    )

    ds_raw = node.results["ds_raw"]
    assert "state" in ds_raw.data_vars
    assert "qubit" in ds_raw["state"].dims
    assert set(map(str, ds_raw.qubit.values.tolist())) == set(QUBIT_NAMES)

    assert "ds_fit" in node.results
    assert "fit_results" in node.results
    fit_ok = node.results["fit_results"][SUCCESS_QUBIT]
    fit_bad = node.results["fit_results"][FAILING_QUBIT]
    assert fit_ok["success"], f"Power-Rabi fit should succeed for {SUCCESS_QUBIT}, got: {fit_ok}"
    assert not fit_bad["success"], f"Power-Rabi fit should fail for {FAILING_QUBIT}, got: {fit_bad}"
    assert node.outcomes[SUCCESS_QUBIT] == "successful"
    assert node.outcomes[FAILING_QUBIT] == "failed"

    opt_amp = float(fit_ok["opt_amp"])
    assert 0.05 < opt_amp < 1.95, f"Expected opt_amp in [0.05, 1.95], got {opt_amp:.4f}"

    omega = float(fit_ok["rabi_frequency"])
    gamma = float(fit_ok["decay_rate"])
    assert np.isfinite(omega) and omega > 0
    assert np.isfinite(gamma)

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert {"rabi", "fft"}.issubset(figures.keys())
    for key in ("rabi", "fft"):
        assert isinstance(figures[key], Figure)
        assert len(figures[key].axes) == 2

    x180 = node.machine.qubits[SUCCESS_QUBIT].macros["x180"]
    assert float(x180.pi_pulse.amplitude) > 0

    baselines = node.namespace["_analysis_test_baselines"][FAILING_QUBIT]
    assert snapshot_qubit_calibration(node.machine.qubits[FAILING_QUBIT]) == baselines
