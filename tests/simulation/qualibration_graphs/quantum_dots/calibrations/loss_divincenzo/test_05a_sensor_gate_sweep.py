from __future__ import annotations
import pytest


NODE_NAME = "05a_sensor_gate_sweep_opx"


@pytest.mark.simulation
def test_sensor_gate_sweep_simulation(simulation_runner):
    """Run simulation and generate artifacts for resonator_spectroscopy."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "num_shots": 10,
            "simulation_duration_ns": 15_000,
            "timeout": 30,  # also in parameters
            "offset_step": 0.05,
        },
    )
