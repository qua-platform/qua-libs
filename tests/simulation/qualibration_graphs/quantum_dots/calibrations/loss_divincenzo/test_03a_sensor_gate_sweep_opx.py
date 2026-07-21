from __future__ import annotations
import pytest

NODE_NAME = "03a_sensor_gate_sweep_opx"


@pytest.mark.simulation
def test_sensor_gate_sweep_opx(simulation_runner):
    """Run simulation and generate artifacts for sensor_gate_sweep_opx."""
    simulation_runner(
        node_name=NODE_NAME,
        param_overrides={
            "offset_min" : -0.4,
            "offset_max" : 0.4,
            "offset_step" : 0.2,
            "peak_fit_side" : "right",
        },
    )
