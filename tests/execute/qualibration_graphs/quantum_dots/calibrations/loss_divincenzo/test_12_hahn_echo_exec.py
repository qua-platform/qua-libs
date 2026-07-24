"""Execute test for 12_hahn_echo."""

from __future__ import annotations

import pytest

NODE_NAME = "12_hahn_echo"


@pytest.mark.execute
def test_hahn_echo_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for Hahn echo parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "qubits": ["q1", "q2"],
            "tau_min": 16,
            "tau_max": 10_000,
            "tau_step": 100,
        },
    )
