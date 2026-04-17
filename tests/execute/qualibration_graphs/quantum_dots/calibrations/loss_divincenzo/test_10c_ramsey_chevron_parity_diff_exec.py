"""Execute test for 10c_ramsey_chevron_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "10c_ramsey_chevron_parity_diff"


@pytest.mark.execute
def test_ramsey_chevron_parity_diff_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for Ramsey chevron parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "detuning_span_in_mhz": 2.0,
            "detuning_step_in_mhz": 1.0,
            "min_wait_time_in_ns": 16,
            "max_wait_time_in_ns": 64,
            "wait_time_num_points": 3,
            "log_or_linear_sweep": "linear",
        },
    )
