"""Simulation test for 10b_ramsey_detuning_parity_diff."""

from __future__ import annotations

import pytest

NODE_NAME = "10b_ramsey_detuning_parity_diff"


@pytest.mark.simulation
def test_ramsey_detuning_parity_diff_simulation(simulation_runner):
    """Run simulation and generate artifacts for ramsey detuning parity diff."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "timeout": 30,
            "detuning_span_in_mhz": 2.0,
            "detuning_step_in_mhz": 0.5,
            "idle_time_ns": 100,
        },
    )
