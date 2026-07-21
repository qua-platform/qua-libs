"""Execute test for 17_geometric_cz_error_amplification."""

from __future__ import annotations

import pytest

NODE_NAME = "17_geometric_cz_error_amplification"
PAIR_NAME = "q1_q2"


@pytest.mark.execute
def test_geometric_cz_error_amplification_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for CZ error amplification."""
    execute_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 1,
            "exchange_amplitude_center": 0.2,
            "exchange_amplitude_span": 0.01,
            "exchange_amplitude_step": 0.01,
            "max_num_cphase_gates": 4,
            "simulation_duration_ns": 40_000,
            "timeout": 300,
        },
    )
