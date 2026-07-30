"""Execute test for 11c_ramsey_chevron."""

from __future__ import annotations

import pytest

NODE_NAME = "11c_ramsey_chevron"


@pytest.mark.execute
def test_ramsey_chevron_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for Ramsey chevron parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "qubits": ["q1", "q2"],
            "detuning_span_in_mhz": 2.0,
            "detuning_step_in_mhz": 0.02,
            "min_wait_time_in_ns": 16,
            "max_wait_time_in_ns": 9_916,
            "wait_time_num_points": 100,
            "log_or_linear_sweep": "linear",
        },
    )
