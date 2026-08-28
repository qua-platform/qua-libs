"""Execute test for 10b_time_rabi_chevron."""

from __future__ import annotations

import pytest

NODE_NAME = "10b_time_rabi_chevron"


@pytest.mark.execute
def test_time_rabi_chevron_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for time Rabi chevron."""
    execute_runner(
        node_name=NODE_NAME,
    )
