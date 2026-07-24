"""Simulation test for 04b_bias_tee_filters.

Runs the frequency-sweep bias tee characterization node in QM-simulation
mode so the QUA program is compiled and the analog waveform is rendered for
inspection, then saves the simulated-samples figure + a README under
``tests/simulation/artifacts``.

Requires QM host configuration; will skip if no cluster is reachable.
"""

from __future__ import annotations

import pytest

NODE_NAME = "04b_bias_tee_filters"


@pytest.mark.simulation
def test_bias_tee_filters_simulation(simulation_runner):
    """Compile and simulate the bias tee frequency-sweep QUA program."""
    simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "elements": ["virtual_dot_1"],
            "sensor_names": ["virtual_sensor_1"],
            "num_shots": 100,
            "square_wave_frequency_start_MHz": 1,
            "square_wave_frequency_stop_MHz": 5,
            "square_wave_frequency_step_MHz": 0.5,
            "square_wave_amplitude": 0.05,
            "simulation_duration_ns": 100_000,
            "timeout": 120,
        },
    )
