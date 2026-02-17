"""Simulation test for 10a_ramsey_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "10a_ramsey_parity_diff"


@pytest.mark.simulation
def test_ramsey_parity_diff_simulation(simulation_runner):
    """Run simulation and generate artifacts for ramsey parity diff."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "timeout": 120,
            "min_wait_time_in_ns": 16,
            "max_wait_time_in_ns": 2000,
            "wait_time_num_points": 3,
            "simulation_duration_ns": 20_000,
            "log_or_linear_sweep": "linear",
        },
    )
