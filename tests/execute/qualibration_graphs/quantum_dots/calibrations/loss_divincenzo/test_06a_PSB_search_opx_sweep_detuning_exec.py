"""Execute test for 06a_PSB_search_opx_sweep_detuning."""

from __future__ import annotations

import pytest

NODE_NAME = "06a_PSB_search_opx_sweep_detuning"
PAIR_NAME = "q1_q2"


@pytest.mark.execute
def test_psb_search_sweep_detuning_execute(execute_runner):
    """Run full execute pipeline and generate artifacts for PSB search sweep detuning."""
    execute_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 2,
            "detuning_min": -0.05,
            "detuning_max": 0.05,
            "detuning_points": 3,
            "ramp_duration": 40,
            "simulation_duration_ns": 40_000,
            "timeout": 120,
        },
    )
