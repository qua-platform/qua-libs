"""Simulation test for 10_T1_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "10_T1_parity_diff"


@pytest.mark.simulation
def test_T1_parity_diff_simulation(simulation_runner):
    """Run simulation and generate artifacts for T1 parity diff."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "num_shots": 10,
            "simulation_duration_ns": 10_000,
            "timeout": 30,
            "tau_min": 500,
        },
    )
