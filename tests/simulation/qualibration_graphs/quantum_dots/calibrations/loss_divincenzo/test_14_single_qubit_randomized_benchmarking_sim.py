"""Simulation test for 14_single_qubit_randomized_benchmarking."""

from __future__ import annotations

import pytest

NODE_NAME = "14_single_qubit_randomized_benchmarking"


@pytest.mark.simulation
def test_single_qubit_rb_simulation(simulation_runner):
    """Compile and simulate a short RB program.

    Uses very small sweep parameters (2 depths, 2 circuits, 2 shots) so
    the simulation completes quickly while still exercising all program
    phases (PPU Clifford generation, gate playback, measurement).
    """
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "num_circuits_per_length": 1,
            "num_shots": 1,
            "max_circuit_depth": 100,
            "log_scale": True,
            "gap_wait_time_in_ns": 1024,
            "simulation_duration_ns": 20_000,
            "timeout": 60,
        },
    )
