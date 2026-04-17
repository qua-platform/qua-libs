"""Execute test for 12_hahn_echo_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "12_hahn_echo_parity_diff"


@pytest.mark.execute
def test_hahn_echo_parity_diff_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for Hahn echo parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "tau_min": 500,
            "tau_step": 500,
        },
    )
