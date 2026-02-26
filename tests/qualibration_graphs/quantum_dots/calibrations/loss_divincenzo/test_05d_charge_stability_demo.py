"""Regression tests for the simulated video-mode charge-stability setup."""

from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("qarray")

ROOT = Path(__file__).resolve().parents[5]
QUANTUM_DOTS_DIR = ROOT / "qualibration_graphs" / "quantum_dots"
if str(QUANTUM_DOTS_DIR) not in sys.path:
    sys.path.insert(0, str(QUANTUM_DOTS_DIR))

from calibration_utils.run_video_mode.simulated_video_mode import (  # noqa: E402
    generate_simulated_video_mode_quam,
    get_simulated_video_mode_base_point,
    setup_simulation,
)


def test_simulated_video_mode_base_point_resolves_virtual_offsets():
    machine = generate_simulated_video_mode_quam()
    gate_set = machine.virtual_gate_sets["main_qpu"]
    base_point = get_simulated_video_mode_base_point(
        {
            "virtual_dot_1": -0.02,
            "virtual_dot_2": -0.01,
        }
    )

    simulator = setup_simulation(
        base_point=base_point,
        gate_set=gate_set,
        dc_set=None,
        sensor_gate_names=["virtual_sensor_1", "virtual_sensor_2"],
    )
    grids = simulator.get_physical_grid(
        "virtual_dot_1",
        "virtual_dot_2",
        np.array([0.0]),
        np.array([0.0]),
    )

    expected = gate_set.resolve_voltages(base_point, allow_extra_entries=True)
    observed = {gate_name: float(grid[0, 0]) for gate_name, grid in zip(simulator.qarray_gate_order, grids)}

    for gate_name in simulator.qarray_gate_order:
        assert observed[gate_name] == pytest.approx(float(expected.get(gate_name, 0.0)))
