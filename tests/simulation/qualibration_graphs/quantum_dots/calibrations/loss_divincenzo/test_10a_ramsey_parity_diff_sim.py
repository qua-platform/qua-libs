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
            "num_shots": 10,
            "simulation_duration_ns": 10_000,
            "timeout": 30,
            "tau_min": 500,
        },
    )
