"""Execute test for 11b_ramsey_detuning."""

from __future__ import annotations

import pytest

NODE_NAME = "11b_ramsey_detuning"


@pytest.mark.execute
def test_ramsey_detuning_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for Ramsey detuning parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "qubits": ["q1", "q2"],
            "detuning_span_in_mhz": 2.0,
            "detuning_step_in_mhz": 0.02,
        },
    )
