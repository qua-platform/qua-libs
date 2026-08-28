"""Analysis test for 09b_power_rabi_error_amplification.

Uses ``generate_simulated_dataset`` from calibration_utils and the shared
``analysis_runner`` fixture. ``q2`` is given structureless data so its fit
fails while ``q1`` succeeds, validating mixed outcomes and selective state updates.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils.power_rabi_error_amplification import generate_simulated_dataset
from analysis_helpers import snapshot_qubit_calibration, with_unfittable_qubit

NODE_NAME = "09b_power_rabi_error_amplification"
QUBIT_NAMES = ["q1", "q2"]
SUCCESS_QUBIT = "q1"
FAILING_QUBIT = "q2"


@pytest.mark.analysis
def test_09b_power_rabi_error_amplification_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot/update with one failing qubit."""
    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=with_unfittable_qubit(generate_simulated_dataset, FAILING_QUBIT),
        param_overrides={
            "num_shots": 8,
            "qubits": QUBIT_NAMES,
        },
    )

    ds_raw = node.results["ds_raw"]
    assert "state" in ds_raw.data_vars
    assert set(ds_raw["state"].dims) >= {"qubit", "n_pulses", "amp_prefactor"}
    assert set(map(str, ds_raw.qubit.values.tolist())) == set(QUBIT_NAMES)

    fit_ok = node.results["fit_results"][SUCCESS_QUBIT]
    fit_bad = node.results["fit_results"][FAILING_QUBIT]
    assert fit_ok["success"], f"Error-amplification fit should succeed for {SUCCESS_QUBIT}, got: {fit_ok}"
    assert not fit_bad["success"], f"Error-amplification fit should fail for {FAILING_QUBIT}, got: {fit_bad}"
    assert node.outcomes[SUCCESS_QUBIT] == "successful"
    assert node.outcomes[FAILING_QUBIT] == "failed"

    opt_amp = float(fit_ok["opt_amp"])
    assert node.parameters.min_amp_factor < opt_amp < node.parameters.max_amp_factor

    omega = float(fit_ok["rabi_frequency"])
    gamma = float(fit_ok["decay_rate"])
    assert np.isfinite(omega) and omega > 0
    assert np.isfinite(gamma)

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert {"heatmap", "resonance"}.issubset(figures.keys())
    for key in ("heatmap", "resonance"):
        assert isinstance(figures[key], Figure)
        assert len(figures[key].axes) == 2

    x180 = node.machine.qubits[SUCCESS_QUBIT].macros["x180"]
    assert float(x180.pi_pulse.amplitude) > 0

    baselines = node.namespace["_analysis_test_baselines"][FAILING_QUBIT]
    assert snapshot_qubit_calibration(node.machine.qubits[FAILING_QUBIT]) == baselines
