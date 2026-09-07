from __future__ import annotations

import pytest

NODE_NAME = "10b_time_rabi_chevron"


@pytest.mark.simulation
def test_time_rabi_chevron_simulation(simulation_runner):
    """Run simulation and generate artifacts for time Rabi chevron."""
    simulation_runner(
        node_name=NODE_NAME,
    )
