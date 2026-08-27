"""Execute test for ``03b_sensor_gate_sweep_dac``."""

from __future__ import annotations

import pytest

NODE_NAME = "03b_sensor_gate_sweep_dac"

SENSOR_GATE_SWEEP_EXEC_PARAMS = {
    "num_shots": 2,
    "offset_min": -0.1,
    "offset_max": 0.1,
    "offset_step": 0.1,
    "timeout": 500,
    "dac_settling_time_s": 0.01,
}


@pytest.mark.execute
def test_03b_sensor_gate_sweep_dac_execute(execute_runner):
    """Run the full DAC sensor gate sweep on real hardware."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides=SENSOR_GATE_SWEEP_EXEC_PARAMS,
        apply_small_sweep=False,
    )
