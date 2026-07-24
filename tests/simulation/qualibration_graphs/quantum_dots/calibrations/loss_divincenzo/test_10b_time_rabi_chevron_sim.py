"""Simulation test for 10b_time_rabi_chevron."""

from __future__ import annotations

import pytest
from conftest import (
    append_area_to_readme,
    assert_balanced_analog_means_if_strict,
    compute_area_under_curve,
)

NODE_NAME = "10b_time_rabi_chevron"


@pytest.mark.simulation
def test_time_rabi_chevron_simulation(simulation_runner):
    """Run simulation, verify area under curve is near zero for all channels."""
    result = simulation_runner(
        node_name=NODE_NAME,
        param_overrides={},
    )
    if result is None:
        pytest.skip("simulation_runner did not return samples")

    samples, artifacts_dir = result
    if samples is None:
        pytest.skip("No simulated samples captured")

    areas = compute_area_under_curve(samples)
    append_area_to_readme(artifacts_dir, areas)

    assert_balanced_analog_means_if_strict(areas)
