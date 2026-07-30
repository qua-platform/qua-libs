"""Execute test for 01_optimize_measurement_fidelity (CMA-ES)."""

from __future__ import annotations

import pytest

NODE_NAME = "01_optimize_measurement_fidelity"
PAIR_NAME = "q1_q2"


@pytest.mark.execute
def test_optimize_measurement_fidelity_execute(execute_runner):
    """Run the full CMA-ES optimisation pipeline on real hardware.

    Uses a small population and few generations to keep the test fast.
    """
    execute_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 50,
            "population_size": 4,
            "max_generations": 3,
            "sigma0": 0.01,
            "detuning_initial": 0.0,
            "detuning_min": -0.05,
            "detuning_max": 0.05,
            "ramp_duration_initial": 100,
            "ramp_duration_min": 16,
            "ramp_duration_max": 500,
            "timeout": 600,
        },
    )
