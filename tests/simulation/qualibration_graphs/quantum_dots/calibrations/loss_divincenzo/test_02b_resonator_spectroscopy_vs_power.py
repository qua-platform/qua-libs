from __future__ import annotations
import pytest

NODE_NAME = "02b_resonator_spectroscopy_vs_power"


@pytest.mark.simulation
def test_resonator_spectroscopy_simulation(simulation_runner):
    """Run simulation and generate artifacts for resonator_spectroscopy."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "timeout": 30,  # also in parameters
        },
    )
