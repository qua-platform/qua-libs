"""Execute test for 09a_power_rabi."""

from __future__ import annotations

import pytest

NODE_NAME = "09a_power_rabi"


@pytest.mark.execute
def test_power_rabi_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for power Rabi."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "qubits": ["q1", "q2"],
            "min_amp_factor": 0.01,
            "max_amp_factor": 1.99,
            "amp_factor_step": 0.02,
        },
    )
