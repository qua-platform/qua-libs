"""Execute test for 17a_iswap_error_amplification."""

from __future__ import annotations

import pytest

NODE_NAME = "17a_iswap_error_amplification"
PAIR_NAME = "q1_q2"


@pytest.mark.execute
def test_iswap_error_amplification_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for residual iSWAP amplification."""
    execute_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 1,
            "exchange_amplitude": 0.2,
            "exchange_duration_in_ns": 64,
            "num_cycle_repetitions": [0, 1, 2],
            "simulation_duration_ns": 40_000,
            "timeout": 300,
        },
    )
