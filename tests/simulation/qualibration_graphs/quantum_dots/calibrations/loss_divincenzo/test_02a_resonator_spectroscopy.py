from __future__ import annotations
import pytest


NODE_NAME = "02a_resonator_spectroscopy"


@pytest.mark.simulation
def test_resonator_spectroscopy_simulation(simulation_runner):
    """Run simulation and generate artifacts for resonator_spectroscopy."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
    "num_shots": 10,
    "simulation_duration_ns": 10_000,
    "timeout": 30,  # also in parameters
},
)