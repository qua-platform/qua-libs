"""Simulation test for 18_cz_phase_compensation."""

from __future__ import annotations

import pytest

from .conftest import (
    append_area_to_readme,
    assert_balanced_analog_means_if_strict,
    compute_area_under_curve,
)

NODE_NAME = "18_cz_phase_compensation"
PAIR_NAME = "q1_q2"


@pytest.mark.simulation
def test_cz_phase_compensation_simulation(simulation_runner):
    """Compile and simulate the CZ phase compensation QUA program, verify voltage balance."""
    result = simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 1,
            "num_frames": 3,
            "simulation_duration_ns": 20_000,
            "timeout": 500,
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
