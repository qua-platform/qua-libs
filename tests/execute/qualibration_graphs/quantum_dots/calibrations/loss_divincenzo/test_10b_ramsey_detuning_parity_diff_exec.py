"""Execute test for 10b_ramsey_detuning_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "10b_ramsey_detuning_parity_dif"


@pytest.mark.execute
def test_ramsey_detuning_parity_diff_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for Ramsey detuning parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "detuning_span_in_mhz": 2.0,
            "detuning_step_in_mhz": 0.5,
            "idle_time_ns": 100,
        },
    )
