"""Simulation test for 01_optimize_measurement_fidelity (CMA-ES)."""

from __future__ import annotations

import pytest
from conftest import (
    append_area_to_readme,
    assert_balanced_analog_means_if_strict,
    compute_area_under_curve,
)

NODE_NAME = "01_optimize_measurement_fidelity"
PAIR_NAME = "q1_q2"


@pytest.mark.simulation
def test_optimize_measurement_fidelity_simulation(simulation_runner):
    """Compile and simulate the CMA-ES measurement fidelity QUA program.

    Only verifies that the program compiles and produces balanced analog
    waveforms (no CMA-ES iterations are run during simulation).
    """
    result = simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 1,
            "population_size": 2,
            "simulation_duration_ns": 40_000,
            "timeout": 300,
        },
    )
    if result is None:
        pytest.skip("simulation_runner did not return samples")

    samples, artifacts_dir = result
    if samples is None:
        pytest.skip("No simulated samples captured")

    areas = compute_area_under_curve(samples)
    append_area_to_readme(artifacts_dir, areas)

    assert_balanced_analog_means_if_strict(areas)
