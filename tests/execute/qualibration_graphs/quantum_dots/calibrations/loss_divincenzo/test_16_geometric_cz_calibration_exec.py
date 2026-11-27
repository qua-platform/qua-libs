"""Execute test for 16_geometric_cz_calibration."""

from __future__ import annotations

import pytest

NODE_NAME = "16_geometric_cz_calibration"
PAIR_NAME = "q1_q2"


@pytest.mark.execute
def test_geometric_cz_calibration_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for geometric CZ calibration."""
    execute_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 1,
            "min_exchange_amplitude": 0.0,
            "max_exchange_amplitude": 0.5,
            "amplitude_step": 0.05,
            "min_exchange_duration_in_ns": 16,
            "max_exchange_duration_in_ns": 500,
            "duration_step_in_ns": 20,
            "simulation_duration_ns": 40_000,
            "timeout": 300,
        },
    )
