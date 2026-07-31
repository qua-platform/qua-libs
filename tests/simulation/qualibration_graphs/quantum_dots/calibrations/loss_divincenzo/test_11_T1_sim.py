"""Simulation test for 11_T1."""

from __future__ import annotations

import pytest
from conftest import (
    append_area_to_readme,
    assert_balanced_analog_means_if_strict,
    compute_area_under_curve,
)

NODE_NAME = "11_T1"


@pytest.mark.simulation
def test_T1_simulation(simulation_runner):
    """Run simulation, verify area under curve is near zero for all channels."""
    result = simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "tau_min": 500,
            "tau_max": 5000,
            "tau_step": 3000,
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
