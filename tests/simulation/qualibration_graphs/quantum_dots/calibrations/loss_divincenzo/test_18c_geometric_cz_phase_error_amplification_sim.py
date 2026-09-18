"""Simulation test for 18c_geometric_cz_phase_error_amplification."""

from __future__ import annotations

import pytest

from .conftest import (
    append_area_to_readme,
    assert_balanced_analog_means_if_strict,
    compute_area_under_curve,
)

NODE_NAME = "18c_geometric_cz_phase_error_amplification"
PAIR_NAME = "q1_q2"


@pytest.mark.simulation
def test_geometric_cz_phase_error_amplification_simulation(simulation_runner):
    """Compile and simulate the CZ phase error-amplification QUA program, verify voltage balance."""
    result = simulation_runner(
        node_name=NODE_NAME,
        apply_small_sweep=False,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 1,
            "exchange_amplitude_center": 0.2,
            "max_num_cphase_gates": 4,
            "num_phases": 3,
            "simulation_duration_ns": 30_000,
            "timeout": 500,
        },
    )
    if result is None:
        pytest.skip("simulation_runner did not return samples")

    samples, artifacts_dir = result
    if samples is None:
        pytest.skip("No simulated samples captured")

    areas = compute_area_under_curve(samples)
    append_area_to_readme(artifacts_dir, areas)

    assert_balanced_analog_means_if_strict(areas)
