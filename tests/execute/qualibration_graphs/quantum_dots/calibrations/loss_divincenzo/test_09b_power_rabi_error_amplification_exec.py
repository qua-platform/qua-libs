"""Execute test for 09b_power_rabi_error_amplification."""

from __future__ import annotations

import pytest

NODE_NAME = "09b_power_rabi_error_amplification"


@pytest.mark.execute
def test_power_rabi_error_amplification_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for error-amplified power Rabi."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "qubits": ["q1", "q2"],
            "min_amp_factor": 0.01,
            "max_amp_factor": 1.99,
            "amp_factor_step": 0.02,
            "max_n_pulses": 20,
        },
    )
