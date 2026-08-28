"""Analysis test for 10b_time_rabi_chevron.

Uses ``generate_simulated_dataset`` from calibration_utils and the shared
``analysis_runner`` fixture. ``q2`` is given structureless data so its fit
fails while ``q1`` succeeds, validating mixed outcomes and selective state updates.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils.time_rabi_chevron import generate_simulated_dataset
from analysis_helpers import snapshot_qubit_calibration, with_unfittable_qubit

NODE_NAME = "10b_time_rabi_chevron"
QUBIT_NAMES = ["q1", "q2"]
SUCCESS_QUBIT = "q1"
FAILING_QUBIT = "q2"


@pytest.mark.analysis
def test_10b_time_rabi_chevron_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot/update on synthetic Rabi-chevron data with one failing qubit."""
    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=with_unfittable_qubit(generate_simulated_dataset, FAILING_QUBIT),
        param_overrides={
            "num_shots": 8,
            "min_wait_time_in_ns": 16,
            "max_wait_time_in_ns": 800,
            "time_step_in_ns": 8,
            "frequency_span_in_mhz": 5.0,
            "frequency_step_in_mhz": 0.5,
            "qubits": QUBIT_NAMES,
        },
    )

    ds_raw = node.results["ds_raw"]
    assert "state" in ds_raw.data_vars
    assert set(ds_raw["state"].dims) >= {"qubit", "detuning", "pulse_duration"}
    assert set(map(str, ds_raw.qubit.values.tolist())) == set(QUBIT_NAMES)

    fit_ok = node.results["fit_results"][SUCCESS_QUBIT]
    fit_bad = node.results["fit_results"][FAILING_QUBIT]
    assert fit_ok["success"], f"Rabi-chevron fit should succeed for {SUCCESS_QUBIT}, got: {fit_ok}"
    assert not fit_bad["success"], f"Rabi-chevron fit should fail for {FAILING_QUBIT}, got: {fit_bad}"
    assert node.outcomes[SUCCESS_QUBIT] == "successful"
    assert node.outcomes[FAILING_QUBIT] == "failed"

    t_pi = float(fit_ok["optimal_duration"])
    assert 16 <= t_pi <= 800, f"Expected t_pi within sweep range, got {t_pi:.0f} ns"

    f_res = float(fit_ok["optimal_frequency"])
    assert np.isfinite(f_res)

    gamma = float(fit_ok["decay_rate"])
    assert np.isfinite(gamma)

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert {"chevron", "fft_2d", "diagnostics"}.issubset(figures.keys())
    for key in ("chevron", "fft_2d", "diagnostics"):
        assert isinstance(figures[key], Figure)
        assert len(figures[key].axes) == 2

    qubit = node.machine.qubits[SUCCESS_QUBIT]
    updated_duration = float(qubit.macros["x180"].pulse.length)
    assert np.isclose(updated_duration, t_pi, rtol=0.0, atol=1e-6)

    baselines = node.namespace["_analysis_test_baselines"][FAILING_QUBIT]
    assert snapshot_qubit_calibration(node.machine.qubits[FAILING_QUBIT]) == baselines
