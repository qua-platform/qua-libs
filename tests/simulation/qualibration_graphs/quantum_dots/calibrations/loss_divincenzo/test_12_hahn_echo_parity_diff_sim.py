"""Simulation test for 12_hahn_echo_parity_diff.

Verifies that the QUA program for the Hahn echo sequence
(π/2 – τ – π – τ – π/2) compiles and simulates correctly.
"""

from __future__ import annotations

import pytest

NODE_NAME = "12_hahn_echo_parity_diff"


@pytest.mark.simulation
def test_hahn_echo_parity_diff_simulation(simulation_runner):
    """Run simulation and generate artifacts for Hahn echo parity diff."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "num_shots": 1,
            "simulation_duration_ns": 30_000,
            "timeout": 30,
            "tau_min": 500,
            "tau_step": 500,
        },
    )
