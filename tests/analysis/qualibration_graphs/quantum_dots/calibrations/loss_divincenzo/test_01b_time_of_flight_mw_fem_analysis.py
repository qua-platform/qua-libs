"""Analysis test for 01b_time_of_flight_mw_fem.

Uses the node package ``generate_simulated_dataset`` (IQ ADC traces with a known
propagation delay) and runs analyse / plot / update via ``analysis_runner``.
"""

from __future__ import annotations

import pytest
from matplotlib.figure import Figure

from calibration_utils.time_of_flight_mw import generate_simulated_dataset
from calibration_utils.time_of_flight_mw.simulated_data_generator import SIMULATED_DELAY_NS

NODE_NAME = "01b_time_of_flight_mw_fem"
SENSOR_NAME = "virtual_sensor_1"
READOUT_LENGTH_NS = 1000


@pytest.mark.analysis
def test_01b_time_of_flight_mw_fem_analysis(analysis_runner):
    """Simulated MW IQ ADC traces should recover TOF delay and figures."""
    node = analysis_runner(
        node_name=NODE_NAME,
        simulated_data_generator=generate_simulated_dataset,
        analyse_qubits=[],
        param_overrides={
            "num_shots": 4,
            "sensor_names": [SENSOR_NAME],
            "readout_length_in_ns": READOUT_LENGTH_NS,
            "time_of_flight_in_ns": 28,
        },
    )

    assert "ds_fit" in node.results
    assert "fit_results" in node.results
    assert SENSOR_NAME in node.results["fit_results"]

    fit = node.results["fit_results"][SENSOR_NAME]
    assert fit["success"], f"TOF fit should succeed, got: {fit}"

    tof_to_add = int(fit["tof_to_add"])
    assert (
        abs(tof_to_add - SIMULATED_DELAY_NS) <= 8
    ), f"Expected tof_to_add near {SIMULATED_DELAY_NS} ns, got {tof_to_add} ns"

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert {"single_run", "averaged_run"}.issubset(figures.keys())
    assert isinstance(figures["single_run"], Figure)
    assert isinstance(figures["averaged_run"], Figure)

    updated_tof = int(node.machine.sensor_dots[SENSOR_NAME].readout_resonator.time_of_flight)
    expected_tof = 28 + tof_to_add
    assert updated_tof == expected_tof, f"Expected TOF {expected_tof}, got {updated_tof}"
