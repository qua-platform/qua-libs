"""Simulation test for 09b_power_rabi_error_amplification."""

from __future__ import annotations

import pytest
from conftest import (
    append_area_to_readme,
    assert_balanced_analog_means_if_strict,
    compute_area_under_curve,
)

NODE_NAME = "09b_power_rabi_error_amplification"


@pytest.mark.simulation
def test_power_rabi_error_amplification_simulation(simulation_runner):
    """Run simulation, verify area under curve is near zero for all channels."""
    result = simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "min_amp_factor": 1.0,
            "amp_factor_step": 0.5,
            # np.arange(2, max_n_pulses, 2) -> [2, 4] for fast simulation
            "max_n_pulses": 6,
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
