"""Execute test for 10_T1_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "10_T1_parity_diff"


@pytest.mark.execute
def test_T1_parity_diff_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for T1 parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "tau_min": 50,
            "tau_step": 3000,
        },
    )
