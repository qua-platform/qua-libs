"""Execute test for 09b_time_rabi_chevron_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "09b_time_rabi_chevron_parity_diff"


@pytest.mark.execute
def test_time_rabi_chevron_parity_diff_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for time Rabi chevron parity diff."""
    execute_runner(
        node_name=NODE_NAME,
    )
