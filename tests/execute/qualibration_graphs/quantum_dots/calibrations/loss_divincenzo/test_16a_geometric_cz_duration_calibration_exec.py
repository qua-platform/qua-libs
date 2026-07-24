"""Execute test for 16a_geometric_cz_duration_calibration."""

from __future__ import annotations

import pytest

NODE_NAME = "16a_geometric_cz_duration_calibration"
PAIR_NAME = "q1_q2"


@pytest.mark.execute
def test_geometric_cz_duration_calibration_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for CZ duration calibration."""
    execute_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 1,
            "exchange_amplitude": 0.3,
            "min_exchange_duration_in_ns": 16,
            "max_exchange_duration_in_ns": 800,
            "duration_step_in_ns": 40,
            "simulation_duration_ns": 40_000,
            "timeout": 300,
        },
    )
