"""Simulation test for ``14a_crot_spectroscopy_parity_diff``.

Runs the CROT spectroscopy node in QM-simulation mode so the QUA program
is compiled and the analog waveform is rendered for inspection, then saves
the simulated-samples figure + a README under ``tests/simulation/artifacts``.

The analysis / plotting / state-update actions are skipped because the node
exits after the simulate stage when ``simulate=True``.

Requires QM host configuration; will skip if no cluster is reachable.
"""

from __future__ import annotations

import pytest

NODE_NAME = "14a_crot_spectroscopy_parity_diff"
PAIR_NAME = "q1_q2"


@pytest.mark.simulation
def test_crot_spectroscopy_parity_diff_simulation(simulation_runner):
    """Compile and simulate the CROT spectroscopy QUA program."""
    simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 1,
            "exchange_min": -0.1,
            "exchange_max": 0.1,
            "exchange_points": 3,
            "esr_frequency_min": 5_000_000_000,
            "esr_frequency_max": 5_500_000_000,
            "esr_frequency_points": 2,
            "duration": 1048,
            "hold_duration": 1048,
            "simulation_duration_ns": 40_000,
            "timeout": 120,
        },
    )
