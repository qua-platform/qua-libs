"""Simulation test for 01b_time_of_flight_mw_fem."""

from __future__ import annotations

import pytest


NODE_NAME = "01b_time_of_flight_mw_fem"


@pytest.mark.simulation
def test_time_of_flight_mw_fem_simulation(simulation_runner, minimal_quam_factory):
    """Run simulation and generate artifacts for MW-FEM time-of-flight."""
    machine = minimal_quam_factory()
    sensor = next(iter(machine.sensor_dots.values()))
    resonator_type = type(sensor.readout_resonator).__name__
    if "MW" not in resonator_type:
        pytest.skip(f"Test machine sensor resonator is {resonator_type}; " "01b requires a MW-FEM readout resonator.")

    simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "num_shots": 1,
            "sensor_names": [sensor.name],
            "readout_length_in_ns": 200,
            "simulation_duration_ns": 40_000,
            "timeout": 120,
        },
    )
