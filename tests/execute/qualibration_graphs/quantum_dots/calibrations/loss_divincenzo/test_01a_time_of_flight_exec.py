"""Execute test for 01a_time_of_flight."""

from __future__ import annotations

import pytest

NODE_NAME = "01a_time_of_flight"


@pytest.mark.execute
def test_time_of_flight_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for time of flight."""
    execute_runner(
        node_name=NODE_NAME,
    )
