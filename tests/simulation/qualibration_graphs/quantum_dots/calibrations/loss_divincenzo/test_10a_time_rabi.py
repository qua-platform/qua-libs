from __future__ import annotations

import pytest

NODE_NAME = "10a_time_rabi"


@pytest.mark.simulation
def test_time_rabi_simulation(simulation_runner):
    """Run simulation and generate artifacts for time Rabi."""
    simulation_runner(
        node_name=NODE_NAME,
    )
