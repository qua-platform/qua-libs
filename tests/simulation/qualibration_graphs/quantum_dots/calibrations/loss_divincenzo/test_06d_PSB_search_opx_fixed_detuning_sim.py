"""Simulation test for ``06d_PSB_search_opx_fixed_detuning``."""

from __future__ import annotations

import pytest

NODE_NAME = "06d_PSB_search_opx_fixed_detuning"
QUBIT_NAME = "q1"


@pytest.mark.simulation
def test_06d_psb_fixed_detuning_simulation(simulation_runner):
    """Compile and simulate the analog waveform only."""
    simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubits": [QUBIT_NAME],
            "num_shots": 4,
            "analysis_model": "gmm",
            "simulation_duration_ns": 40_000,
            "timeout": 120,
        },
    )
