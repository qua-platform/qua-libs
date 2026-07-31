"""Execute test for 02b_resonator_spectroscopy_vs_power."""

from __future__ import annotations

import pytest

NODE_NAME = "02b_resonator_spectroscopy_vs_power"


@pytest.mark.execute
def test_resonator_spectroscopy_vs_power_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for resonator spectroscopy vs power."""
    execute_runner(
        node_name=NODE_NAME,
    )
