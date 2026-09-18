from __future__ import annotations

import pytest

NODE_NAME = "09a_power_rabi"


@pytest.mark.simulation
def test_power_rabi_simulation(simulation_runner):
    """Run simulation and generate artifacts for power Rabi."""
    simulation_runner(
        node_name=NODE_NAME,
    )
