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
            "num_shots": 1,
            "simulation_duration_ns": 30_000,
            "timeout": 100,
            "tau_min": 50,
            "tau_step": 3000,
        },
    )
