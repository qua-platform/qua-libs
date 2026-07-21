"""Simulation test for 07_init_ramp_rate_calibration."""

from __future__ import annotations

import pytest
from conftest import (
    append_area_to_readme,
    assert_balanced_analog_means_if_strict,
    compute_area_under_curve,
)

NODE_NAME = "07_init_ramp_rate_calibration"
PAIR_NAME = "q1_q2"


@pytest.mark.simulation
def test_init_ramp_rate_calibration_simulation(simulation_runner):
    """Compile and simulate the analog waveform; verify balanced means."""
    result = simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 1,
            "ramp_duration_min": 16,
            "ramp_duration_max": 1000,
            "ramp_duration_step": 500,
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
