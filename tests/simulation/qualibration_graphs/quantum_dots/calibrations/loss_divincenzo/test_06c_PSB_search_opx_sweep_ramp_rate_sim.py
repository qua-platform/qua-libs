"""Simulation test for ``06c_PSB_search_opx_sweep_ramp_rate``."""

from __future__ import annotations

import pytest

NODE_NAME = "06c_PSB_search_opx_sweep_ramp_rate"
PAIR_NAME = "q1_q2"


@pytest.mark.simulation
def test_06c_psb_sweep_ramp_rate_simulation(simulation_runner):
    """Compile and simulate the analog waveform only."""
    simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 2,
            "ramp_duration_min": 16,
            "ramp_duration_max": 64,
            "ramp_duration_step": 16,
            "simulation_duration_ns": 40_000,
            "timeout": 120,
        },
    )
