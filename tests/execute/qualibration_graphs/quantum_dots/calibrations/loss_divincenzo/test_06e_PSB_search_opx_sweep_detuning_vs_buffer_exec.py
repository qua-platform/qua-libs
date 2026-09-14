"""Execute test for ``06e_PSB_search_opx_sweep_detuning_vs_buffer``."""

from __future__ import annotations

import pytest

NODE_NAME = "06e_PSB_search_opx_sweep_detuning_vs_buffer"
PAIR_NAME = "q1_q2"


@pytest.mark.execute
def test_06e_psb_sweep_detuning_vs_buffer_execute(execute_runner):
    """Run the full execute pipeline and generate artifacts."""
    execute_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 2,
            "detuning_min": -0.05,
            "detuning_max": 0.05,
            "detuning_points": 3,
            "buffer_duration_min": 16,
            "buffer_duration_max": 64,
            "buffer_duration_step": 16,
            "ramp_duration": 40,
            "simulation_duration_ns": 40_000,
            "timeout": 120,
        },
    )
