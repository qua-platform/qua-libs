"""Analysis test for 13_xy8.

Uses ``generate_simulated_dataset`` from calibration_utils and the shared
``analysis_runner`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils.xy8 import generate_simulated_dataset
from calibration_utils.xy8.simulated_data_generator import DEFAULT_T2_XY8_NS

NODE_NAME = "13_xy8"
QUBIT_NAMES = ["q1", "q2"]


@pytest.mark.analysis
def test_13_xy8_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot/update on synthetic XY8 decay data."""
    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=generate_simulated_dataset,
        analyse_qubits=QUBIT_NAMES,
        param_overrides={
            "num_shots": 8,
            "tau_min": 16,
            "tau_max": 6_000,
            "tau_step": 64,
            "sim_noise_std": 0.01,
            "qubits": QUBIT_NAMES,
        },
    )

    ds_raw = node.results["ds_raw"]
    assert "tau" in ds_raw.dims
    for qname in QUBIT_NAMES:
        assert f"p_{qname}" in ds_raw.data_vars
    assert not any(name.startswith("E_") for name in ds_raw.data_vars)

    assert "ds_fit" in node.results
    assert "fit_results" in node.results

    for qname in QUBIT_NAMES:
        fit = node.results["fit_results"][qname]
        assert fit["success"], f"XY8 fit should succeed for {qname}, got: {fit}"

        t2_xy8 = float(fit["T2_xy8"])
        assert t2_xy8 > 0 and np.isfinite(t2_xy8)
        assert (
            abs(t2_xy8 - DEFAULT_T2_XY8_NS) < 0.3 * DEFAULT_T2_XY8_NS
        ), f"{qname} T2_xy8 should be near {DEFAULT_T2_XY8_NS:.0f} ns, got {t2_xy8:.1f} ns"

        assert float(fit["decay_rate"]) > 0
        assert float(fit["amplitude"]) > 0

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert "decay" in figures
    assert isinstance(figures["decay"], Figure)
    assert len(figures["decay"].axes) > 0
