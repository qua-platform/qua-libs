"""Execute test for 02a_resonator_spectroscopy."""

from __future__ import annotations

import pytest

NODE_NAME = "02a_resonator_spectroscopy"


@pytest.mark.execute
def test_resonator_spectroscopy_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for resonator spectroscopy."""
    execute_runner(
        node_name=NODE_NAME,
    )
