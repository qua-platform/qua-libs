"""Execute test for 11_T1."""

from __future__ import annotations

import pytest

NODE_NAME = "11_T1"


@pytest.mark.execute
def test_T1_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for T1 parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "qubits": ["q1", "q2"],
            "tau_min": 16,
            "tau_max": 10_000,
            "tau_step": 100,
        },
    )
