"""Execute test for 13_xy8."""

from __future__ import annotations

import pytest

NODE_NAME = "13_xy8"


@pytest.mark.execute
def test_xy8_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for XY8 parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "qubits": ["q1", "q2"],
            "tau_min": 16,
            "tau_max": 10_000,
            "tau_step": 100,
        },
    )
