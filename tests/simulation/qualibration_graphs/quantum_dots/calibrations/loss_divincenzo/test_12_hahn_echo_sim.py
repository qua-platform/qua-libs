"""Simulation test for 12_hahn_echo.

Verifies that the QUA program for the Hahn echo sequence
(π/2 – τ – π – τ – π/2) compiles and simulates correctly,
and that all analog channels return to zero (balanced waveforms).
"""

from __future__ import annotations

import pytest
from conftest import (
    append_area_to_readme,
    assert_balanced_analog_means_if_strict,
    compute_area_under_curve,
)

NODE_NAME = "12_hahn_echo"


@pytest.mark.simulation
def test_hahn_echo_simulation(simulation_runner):
    """Run simulation, verify area under curve is near zero for all channels."""
    result = simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "tau_min": 100,
            "tau_max": 1000,
            "tau_step": 500,
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
