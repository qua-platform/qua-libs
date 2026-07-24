"""Simulation test for 05c_charge_state_readout_time_optimization.

Runs the charge-state readout time optimization node in QM-simulation mode
so the QUA program is compiled and the analog waveform is rendered for
inspection, then saves the simulated-samples figure + a README under
``tests/simulation/artifacts``.

Requires QM host configuration; will skip if no cluster is reachable.
"""

from __future__ import annotations

import pytest

NODE_NAME = "05c_charge_state_readout_time_optimization"


@pytest.mark.simulation
def test_charge_state_readout_time_optimization_simulation(simulation_runner):
    """Compile and simulate the charge-state readout time optimization QUA program."""
    simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "quantum_dots": ["virtual_dot_1", "virtual_dot_2"],
            "num_shots": 10,
            "integration_time_start": 100,
            "integration_time_stop": 2000,
            "integration_time_step": 100,
            "wait_time": 4000,
            "simulation_duration_ns": 40_000,
            "timeout": 120,
        },
    )
