"""Analysis test for 10a_time_rabi.

Uses ``generate_simulated_dataset`` from calibration_utils and the shared
``analysis_runner`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils.time_rabi import generate_simulated_dataset

NODE_NAME = "10a_time_rabi"
QUBIT_NAMES = ["q1", "q2"]


@pytest.mark.analysis
def test_10a_time_rabi_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot/update on synthetic 1D time-Rabi data."""
    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=generate_simulated_dataset,
        param_overrides={
            "num_shots": 8,
            "min_wait_time_in_ns": 16,
            "max_wait_time_in_ns": 2000,
            "time_step_in_ns": 8,
            "qubits": QUBIT_NAMES,
        },
    )

    ds_raw = node.results["ds_raw"]
    assert "state" in ds_raw.data_vars
    assert set(ds_raw["state"].dims) >= {"qubit", "pulse_duration"}
    assert set(map(str, ds_raw.qubit.values.tolist())) == set(QUBIT_NAMES)

    fit = node.results["fit_results"][QUBIT_NAMES[0]]
    assert fit["success"], f"Time-Rabi fit should succeed, got: {fit}"

    t_pi = float(fit["optimal_duration"])
    assert 16 <= t_pi <= 2000, f"Expected t_pi within sweep range, got {t_pi:.0f} ns"

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

    updated_duration = float(node.machine.qubits[QUBIT_NAMES[0]].macros["x180"].pulse.length)
    assert np.isclose(updated_duration, t_pi, rtol=0.0, atol=1e-6)
