from typing import Dict, List
from qua_dashboards.video_mode.inner_loop_actions.simulators import QarraySimulator

try:
    from qarray import ChargeSensedDotArray
except:
    raise ImportError("qarray is not installed. Please install it using `pip install qarray`.")


def setup_simulation(base_point: Dict[str, float], gate_set, dc_set=None, sensor_gate_names: List[str] = None):
    Cdd = [
        [0.12, 0.08],
        [0.08, 0.13],
    ]
    Cgd = [
        [0.13, 0.00, 0.00, 0.00],
        [0.00, 0.11, 0.00, 0.00],
    ]
    Cds = [
        [0.002, 0.002],
        [0.002, 0.002],
    ]
    Cgs = [
        [0.001, 0.002, 0.100, 0.000],
        [0.001, 0.002, 0.000, 0.100],
    ]
    model = ChargeSensedDotArray(
        Cdd=Cdd,
        Cgd=Cgd,
        Cds=Cds,
        Cgs=Cgs,
        coulomb_peak_width=0.9,
        T=50.0,
        algorithm="default",
        implementation="jax",
    )

    simulator = QarraySimulator(
        gate_set=gate_set,
        dc_set=dc_set,
        model=model,
        sensor_gate_names=sensor_gate_names,
        base_point=base_point,
    )

    return simulator
