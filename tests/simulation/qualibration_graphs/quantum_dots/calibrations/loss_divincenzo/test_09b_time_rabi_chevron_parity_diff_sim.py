"""Simulation test for 09b_time_rabi_chevron_parity_diff."""

from __future__ import annotations

import pytest


NODE_NAME = "09b_time_rabi_chevron_parity_diff"


@pytest.mark.simulation
def test_time_rabi_chevron_parity_diff_simulation(simulation_runner):
    """Run simulation and generate artifacts for time rabi chevron parity diff."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "num_shots": 10,
            "simulation_duration_ns": 10_000,
            "timeout": 30,  # also in parameters
        },
    )
