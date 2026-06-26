"""Execute test for 07_init_ramp_rate_calibration."""

from __future__ import annotations

import pytest

NODE_NAME = "07_init_ramp_rate_calibration"
PAIR_NAME = "q1_q2"


@pytest.mark.execute
def test_init_ramp_rate_calibration_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for init ramp rate calibration."""
    execute_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 10,
            "ramp_duration_min": 16,
            "ramp_duration_max": 2000,
            "ramp_duration_step": 200,
            "timeout": 300,
        },
    )
