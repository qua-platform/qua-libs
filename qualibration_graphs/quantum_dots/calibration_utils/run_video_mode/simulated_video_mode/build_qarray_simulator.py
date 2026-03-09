from typing import Dict, List

import numpy as np
from qua_dashboards.video_mode.inner_loop_actions.simulators import QarraySimulator

try:
    from qarray import ChargeSensedDotArray
except:
    raise ImportError("qarray is not installed. Please install it using `pip install qarray`.")


DEFAULT_SIMULATED_VIDEO_MODE_BASE_POINT = {
    "virtual_dot_1": -20.0e-3,
    "virtual_dot_2": -20.0e-3,
    "virtual_sensor_1": -5.0e-3,
    "virtual_sensor_2": -5.0e-3,
}


def get_simulated_video_mode_base_point(
    overrides: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Return the default virtual-gate center for simulated video mode."""
    base_point = DEFAULT_SIMULATED_VIDEO_MODE_BASE_POINT.copy()
    if overrides is not None:
        base_point.update(overrides)
    return base_point


class SimulatedVideoModeQarraySimulator(QarraySimulator):
    """Resolve virtual-gate centers before constructing the physical qarray grid."""

    def _base_levels(self) -> Dict[str, float]:
        if self.dc_set is not None:
            return dict(getattr(self.dc_set, "_current_levels", {}) or {})
        return dict(self.base_point)

    def _resolved_base_levels(self) -> Dict[str, float]:
        base = self._base_levels()
        resolved = self.gate_set.resolve_voltages(base, allow_extra_entries=True)
        for gate_name in self.qarray_gate_order:
            if gate_name in base and gate_name not in resolved:
                resolved[gate_name] = float(base[gate_name])
        return resolved

    def get_physical_grid(self, x_axis_name, y_axis_name, x_vals, y_vals):
        """Vectorized physical-grid generation with support for virtual-gate centers."""
        base = self._base_levels()
        base_phys = self._resolved_base_levels()

        x_vals = np.asarray(x_vals, float)
        y_vals = np.asarray(y_vals, float)
        x_vals = x_vals - x_vals[len(x_vals) // 2]
        y_vals = y_vals - y_vals[len(y_vals) // 2]

        zero_point = {name: 0.0 for name in base}
        delta_x = self.gate_set.resolve_voltages(
            {**zero_point, x_axis_name: 1.0},
            allow_extra_entries=True,
        )
        delta_y = self.gate_set.resolve_voltages(
            {**zero_point, y_axis_name: 1.0},
            allow_extra_entries=True,
        )

        base_vec = np.array([float(base_phys.get(gate_name, 0.0)) for gate_name in self.qarray_gate_order])
        dx_vec = np.array([float(delta_x.get(gate_name, 0.0)) for gate_name in self.qarray_gate_order])
        dy_vec = np.array([float(delta_y.get(gate_name, 0.0)) for gate_name in self.qarray_gate_order])

        X, Y = np.meshgrid(x_vals, y_vals)
        grids = base_vec[None, None, :] + X[:, :, None] * dx_vec[None, None, :] + Y[:, :, None] * dy_vec[None, None, :]
        return [grids[:, :, i] for i in range(len(self.qarray_gate_order))]


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

    simulator = SimulatedVideoModeQarraySimulator(
        gate_set=gate_set,
        dc_set=dc_set,
        model=model,
        sensor_gate_names=sensor_gate_names,
        base_point=get_simulated_video_mode_base_point(base_point),
    )

    return simulator
