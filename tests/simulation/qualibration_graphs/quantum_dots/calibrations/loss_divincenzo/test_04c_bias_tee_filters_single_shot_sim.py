"""Simulation test for 04c_bias_tee_filters_single_shot.

Runs the single-shot (sliced demodulation) bias tee characterization node
in QM-simulation mode so the QUA program is compiled and the analog waveform
is rendered for inspection, then saves the simulated-samples figure + a
README under ``tests/simulation/artifacts``.

Requires QM host configuration; will skip if no cluster is reachable.
"""

from __future__ import annotations

import pytest

NODE_NAME = "04c_bias_tee_filters_single_shot"


@pytest.mark.simulation
def test_bias_tee_filters_single_shot_simulation(simulation_runner):
    """Compile and simulate the bias tee single-shot QUA program."""
    simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "elements": ["virtual_dot_1"],
            "sensor_names": ["virtual_sensor_1"],
            "num_shots": 100,
            "step_amplitude": 0.05,
            "measurement_time": 4000,
            "integration_time": 100,
            "simulation_duration_ns": 35_000,
            "timeout": 120,
        },
    )
