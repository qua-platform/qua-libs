"""Execute test for ``06b_PSB_search_opx_fixed_detuning_measure_duration``."""

from __future__ import annotations

import pytest

NODE_NAME = "06b_PSB_search_opx_fixed_detuning_measure_duration"
PAIR_NAME = "q1_q2"


@pytest.mark.execute
def test_06b_psb_fixed_detuning_measure_duration_execute(execute_runner):
    """Run the full execute pipeline and generate artifacts."""
    execute_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 2,
            "readout_length_min": 100,
            "readout_length_max": 300,
            "readout_length_points": 3,
            "ramp_duration": 40,
            "simulation_duration_ns": 40_000,
            "timeout": 120,
        },
    )
