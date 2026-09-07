"""Execute test for 10a_time_rabi."""

from __future__ import annotations

import pytest

NODE_NAME = "10a_time_rabi"


@pytest.mark.execute
def test_time_rabi_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for time Rabi."""
    execute_runner(
        node_name=NODE_NAME,
    )
