"""Simulation test for 09a_time_rabi_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "09a_time_rabi_parity_diff"


@pytest.mark.simulation
def test_time_rabi_parity_diff_simulation(simulation_runner):
    """Run simulation and generate artifacts for time rabi parity diff."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "time_step_in_ns": 500,
        },
    )
