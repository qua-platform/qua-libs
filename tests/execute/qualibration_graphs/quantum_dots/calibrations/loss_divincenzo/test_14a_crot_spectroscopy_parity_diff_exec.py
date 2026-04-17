"""Execute test for 14a_crot_spectroscopy_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "14a_crot_spectroscopy_parity_diff"
PAIR_NAME = "q1_q2"


@pytest.mark.execute
def test_crot_spectroscopy_parity_diff_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for CROT spectroscopy parity diff."""
    execute_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 1,
            "exchange_min": -0.1,
            "exchange_max": 0.1,
            "exchange_points": 3,
            "esr_frequency_min": 5_000_000_000,
            "esr_frequency_max": 5_500_000_000,
            "esr_frequency_points": 2,
            "duration": 1048,
            "hold_duration": 1048,
            "simulation_duration_ns": 40_000,
            "timeout": 120,
        },
    )
