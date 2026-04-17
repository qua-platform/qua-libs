"""Execute test for 10a_ramsey_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "10a_ramsey_parity_diff"


@pytest.mark.execute
def test_ramsey_parity_diff_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for Ramsey parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "min_wait_time_in_ns": 16,
            "max_wait_time_in_ns": 2000,
            "wait_time_num_points": 2,
            "log_or_linear_sweep": "linear",
        },
    )
