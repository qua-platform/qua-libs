"""Simulation test for 11c_ramsey_chevron."""

from __future__ import annotations

import pytest
from conftest import (
    append_area_to_readme,
    assert_balanced_analog_means_if_strict,
    compute_area_under_curve,
)

NODE_NAME = "11c_ramsey_chevron"


@pytest.mark.simulation
def test_ramsey_chevron_simulation(simulation_runner):
    """Run simulation, verify area under curve is near zero for all channels."""
    result = simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "detuning_span_in_mhz": 2.0,
            "detuning_step_in_mhz": 1.0,
            "min_wait_time_in_ns": 16,
            "max_wait_time_in_ns": 64,
            "wait_time_num_points": 3,
            "log_or_linear_sweep": "linear",
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
