"""Simulation test for ``06a_PSB_search_opx_sweep_detuning``.

Runs the node in QM-simulation mode so the QUA program is compiled and the
analog waveform is rendered for inspection, then saves the simulated-samples
figure + a README under ``tests/simulation/artifacts``. The analysis /
plotting / state-update run_actions are skipped (node.run with simulate=True
exits after the simulate stage).

Requires QM host configuration; will skip if no cluster is reachable (i.e.
not on the VPN).
"""

from __future__ import annotations

import pytest

NODE_NAME = "06a_PSB_search_opx_sweep_detuning"
PAIR_NAME = "q1_q2"


@pytest.mark.simulation
def test_06a_psb_search_sweep_detuning_simulation(simulation_runner):
    """Compile and simulate the analog waveform only."""
    simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,  # 06a manages its own qubit-pair detuning sweep
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 2,
            "detuning_min": -0.05,
            "detuning_max": 0.05,
            "detuning_points": 3,
            "ramp_duration": 40,
            "simulation_duration_ns": 40_000,
            "timeout": 120,
        },
    )
