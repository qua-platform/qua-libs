"""Analysis test for 10b_time_rabi_chevron.

Uses ``generate_simulated_dataset`` from calibration_utils and the shared
``analysis_runner`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils.time_rabi_chevron import generate_simulated_dataset

NODE_NAME = "10b_time_rabi_chevron"
QUBIT_NAME = "q1"


@pytest.mark.analysis
def test_10b_time_rabi_chevron_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot/update on synthetic Rabi-chevron data."""
    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=generate_simulated_dataset,
        param_overrides={
            "num_shots": 8,
            "min_wait_time_in_ns": 16,
            "max_wait_time_in_ns": 800,
            "time_step_in_ns": 8,
            "frequency_span_in_mhz": 5.0,
            "frequency_step_in_mhz": 0.5,
            "qubits": [QUBIT_NAME],
        },
    )

    ds_raw = node.results["ds_raw"]
    assert f"p_{QUBIT_NAME}" in ds_raw.data_vars
    assert set(ds_raw.dims) >= {"detuning", "pulse_duration"}
    assert not any(name.startswith("E_") for name in ds_raw.data_vars)

    fit = node.results["fit_results"][QUBIT_NAME]
    assert fit["success"], f"Rabi-chevron fit should succeed, got: {fit}"

    t_pi = float(fit["optimal_duration"])
    assert 16 <= t_pi <= 800, f"Expected t_pi within sweep range, got {t_pi:.0f} ns"

    f_res = float(fit["optimal_frequency"])
    assert np.isfinite(f_res)

    gamma = float(fit["decay_rate"])
    assert np.isfinite(gamma)

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert {"chevron", "fft_2d", "diagnostics"}.issubset(figures.keys())
    for key in ("chevron", "fft_2d", "diagnostics"):
        assert isinstance(figures[key], Figure)
        assert len(figures[key].axes) > 0

    qubit = node.machine.qubits[QUBIT_NAME]
    updated_duration = float(qubit.macros["x180"].pulse.length)
    assert np.isclose(updated_duration, t_pi, rtol=0.0, atol=1e-6)
