"""Simulation test for 01_sensor_gate_compensation."""

from __future__ import annotations

import pytest

NODE_NAME = "01_sensor_gate_compensation"


@pytest.mark.simulation
def test_sensor_gate_compensation_simulation(simulation_runner):
    """Run sensor-gate compensation in simulate mode and generate artifacts."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "num_shots": 1,
            "sensor_gate_span": 0.02,
            "sensor_gate_points": 21,
            "device_gate_span": 0.02,
            "device_gate_points": 21,
            "simulation_duration_ns": 20_000,
            "timeout": 30,
            "sensor_device_mapping": {"virtual_sensor_1": ["virtual_dot_1"]},
        },
    )
