"""Simulation test for 02_virtual_plunger_calibration."""

from __future__ import annotations

import pytest

NODE_NAME = "02_virtual_plunger_calibration"


@pytest.mark.simulation
def test_virtual_plunger_calibration_simulation(simulation_runner):
    """Run virtual-plunger calibration in simulate mode and generate artifacts."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "num_shots": 1,
            "plunger_gate_span": 0.02,
            "plunger_gate_points": 21,
            "device_gate_span": 0.02,
            "device_gate_points": 21,
            "simulation_duration_ns": 20_000,
            "timeout": 30,
            "plunger_device_mapping": {"virtual_dot_1": ["virtual_dot_2"]},
        },
    )
