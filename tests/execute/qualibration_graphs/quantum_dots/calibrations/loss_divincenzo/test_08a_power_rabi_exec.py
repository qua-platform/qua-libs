"""Execute test for 08a_power_rabi."""

from __future__ import annotations

import pytest

NODE_NAME = "08a_power_rabi"


@pytest.mark.execute
def test_power_rabi_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for power Rabi."""
    execute_runner(
        node_name=NODE_NAME,
        param_overrides={
            "min_amp_factor": 1.0,
            "amp_factor_step": 0.5,
        },
    )
