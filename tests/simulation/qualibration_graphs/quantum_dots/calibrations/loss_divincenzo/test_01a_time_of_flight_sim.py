"""Simulation test for 01a_time_of_flight."""

from __future__ import annotations

import pytest


NODE_NAME = "01a_time_of_flight"


@pytest.mark.simulation
def test_time_of_flight_simulation(simulation_runner):
    """Run simulation and generate artifacts for time-of-flight."""
    simulation_runner(
        node_name=NODE_NAME,
    )
