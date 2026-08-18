"""Execute test for ``03a_sensor_gate_sweep_opx``."""

from __future__ import annotations

import pytest

NODE_NAME = "03a_sensor_gate_sweep_opx"

SENSOR_GATE_SWEEP_EXEC_PARAMS = {
    "num_shots": 2,
    "offset_min": -0.1,
    "offset_max": 0.1,
    "offset_step": 0.1,
    "timeout": 500,
}


@pytest.mark.execute
def test_03a_sensor_gate_sweep_opx_execute(execute_runner):
    """Run the full OPX sensor gate sweep on real hardware."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides=SENSOR_GATE_SWEEP_EXEC_PARAMS,
        apply_small_sweep=False,
    )
