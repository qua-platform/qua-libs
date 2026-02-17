"""Simulation test for 10c_ramsey_chevron_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "10c_ramsey_chevron_parity_diff"


@pytest.mark.simulation
def test_ramsey_chevron_parity_diff_simulation(simulation_runner):
    """Run simulation and generate artifacts for ramsey chevron parity diff."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "num_shots": 10,
            "simulation_duration_ns": 20_000,
            "timeout": 120,
            "detuning_span_in_mhz": 2.0,
            "detuning_step_in_mhz": 1.0,
            "min_wait_time_in_ns": 16,
            "max_wait_time_in_ns": 64,
            "wait_time_num_points": 3,
            "log_or_linear_sweep": "linear",
        },
    )
