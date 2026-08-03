"""Analysis test for 12_hahn_echo.

Uses ``generate_simulated_dataset`` from calibration_utils and the shared
``analysis_runner`` fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils.hahn_echo import generate_simulated_dataset
from calibration_utils.hahn_echo.simulated_data_generator import DEFAULT_T2_ECHO_NS

NODE_NAME = "12_hahn_echo"
QUBIT_NAMES = ["q1", "q2"]


@pytest.mark.analysis
def test_12_hahn_echo_analysis_and_plot_actions(analysis_runner):
    """Run analyse/plot/update on synthetic Hahn-echo decay data."""
    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=generate_simulated_dataset,
        analyse_qubits=QUBIT_NAMES,
        param_overrides={
            "num_shots": 8,
            "tau_min": 16,
            "tau_max": 3_000,
            "tau_step": 40,
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
        assert fit["success"], f"Hahn-echo fit should succeed for {qname}, got: {fit}"

        t2_echo = float(fit["T2_echo"])
        assert t2_echo > 0 and np.isfinite(t2_echo)
        assert (
            abs(t2_echo - DEFAULT_T2_ECHO_NS) < 0.3 * DEFAULT_T2_ECHO_NS
        ), f"{qname} T2_echo should be near {DEFAULT_T2_ECHO_NS:.0f} ns, got {t2_echo:.1f} ns"

        assert float(fit["decay_rate"]) > 0
        assert float(fit["amplitude"]) > 0

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert "decay" in figures
    assert isinstance(figures["decay"], Figure)
    assert len(figures["decay"].axes) > 0

    for qname in QUBIT_NAMES:
        fitted_t2 = float(node.results["fit_results"][qname]["T2_echo"])
        assert np.isclose(float(node.machine.qubits[qname].T2echo), fitted_t2, rtol=0.0, atol=1e-6)
