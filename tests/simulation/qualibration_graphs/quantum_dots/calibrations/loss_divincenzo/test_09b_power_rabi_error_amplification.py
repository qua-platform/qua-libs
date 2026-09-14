from __future__ import annotations

import pytest

NODE_NAME = "09b_power_rabi_error_amplification"


@pytest.mark.simulation
def test_power_rabi_error_amplification_simulation(simulation_runner):
    """Run simulation and generate artifacts for error-amplified power Rabi."""
    simulation_runner(
        node_name=NODE_NAME,
    )
