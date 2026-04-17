"""Execute test for 09a_time_rabi_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "09a_time_rabi_parity_diff"


@pytest.mark.execute
def test_time_rabi_parity_diff_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for time Rabi parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "time_step_in_ns": 500,
        },
    )
