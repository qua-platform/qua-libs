"""Simulation test for 15_crot_spectroscopy."""

from __future__ import annotations

import pytest
from conftest import (
    append_area_to_readme,
    assert_balanced_analog_means_if_strict,
    compute_area_under_curve,
)

NODE_NAME = "15_crot_spectroscopy"
PAIR_NAME = "q1_q2"


@pytest.mark.simulation
def test_crot_spectroscopy_simulation(simulation_runner):
    """Compile and simulate the CROT spectroscopy QUA program, verify voltage balance."""
    result = simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 1,
            "exchange_min": -0.1,
            "exchange_max": 0.1,
            "exchange_points": 3,
            "esr_frequency_min": 5_000_000_000,
            "esr_frequency_max": 5_500_000_000,
            "esr_frequency_points": 2,
            "duration": 1048,
            "simulation_duration_ns": 40_000,
            "timeout": 120,
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
