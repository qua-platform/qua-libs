"""Simulation test for ``03a_sensor_gate_sweep_opx``."""

from __future__ import annotations

import pytest

NODE_NAME = "03a_sensor_gate_sweep_opx"

SENSOR_GATE_SWEEP_SIM_PARAMS = {
    "num_shots": 1,
    "offset_min": -0.2,
    "offset_max": 0.2,
    "offset_step": 0.2,
    "simulation_duration_ns": 150_000,
    "timeout": 300,
}


@pytest.mark.simulation
def test_03a_sensor_gate_sweep_opx_simulation(simulation_runner):
    """Simulate the OPX sensor gate sweep and write waveform artifacts."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides=SENSOR_GATE_SWEEP_SIM_PARAMS,
        apply_small_sweep=False,
    )
