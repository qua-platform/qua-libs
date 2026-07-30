"""Execute test for 10b_time_rabi_chevron."""

from __future__ import annotations

import pytest

NODE_NAME = "10b_time_rabi_chevron"


@pytest.mark.execute
def test_time_rabi_chevron_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for time Rabi chevron parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "qubits": ["q1", "q2"],
            "min_wait_time_in_ns": 16,
            "max_wait_time_in_ns": 10_000,
            "time_step_in_ns": 100,
            "frequency_span_in_mhz": 2.0,
            "frequency_step_in_mhz": 0.02,
        },
    )
