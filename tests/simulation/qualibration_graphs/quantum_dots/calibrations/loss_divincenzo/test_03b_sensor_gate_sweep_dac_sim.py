"""Simulation test for ``03b_sensor_gate_sweep_dac``."""

from __future__ import annotations

import pytest

NODE_NAME = "03b_sensor_gate_sweep_dac"

SENSOR_GATE_SWEEP_SIM_PARAMS = {
    "num_shots": 1,
    "offset_min": -0.2,
    "offset_max": 0.2,
    "offset_step": 0.2,
    "simulation_duration_ns": 150_000,
    "timeout": 300,
}


@pytest.mark.simulation
def test_03b_sensor_gate_sweep_dac_simulation(simulation_runner):
    """Simulate the DAC-paused sensor gate sweep and write waveform artifacts."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides=SENSOR_GATE_SWEEP_SIM_PARAMS,
        apply_small_sweep=False,
    )
