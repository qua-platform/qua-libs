"""Execute test for 09a_power_rabi."""

from __future__ import annotations

import pytest

NODE_NAME = "09a_power_rabi"


@pytest.mark.execute
def test_power_rabi_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for power Rabi."""
    execute_runner(
        node_name=NODE_NAME,
    )
