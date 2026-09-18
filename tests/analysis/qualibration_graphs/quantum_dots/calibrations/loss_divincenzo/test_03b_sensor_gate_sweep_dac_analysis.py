"""Analysis test for ``03b_sensor_gate_sweep_dac``."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils.sensor_dot.simulated_data_generator import generate_simulated_dataset

from .conftest import ARTIFACTS_BASE, CALIBRATION_LIBRARY_ROOT

NODE_NAME = "03b_sensor_gate_sweep_dac"
SENSOR_NAME = "virtual_sensor_1"
DAC_OFFSET_V = 0.05


def _run_03b_analysis(*, machine, param_overrides: Dict[str, Any]) -> Any:
    from shared_fixtures import (
        apply_param_overrides,
        call_node_action,
        ensure_quam_config_stub,
        load_library_node,
        make_save_analysis_plot,
        patch_action_manager_register_only,
        reimport_node_to_register_actions,
    )

    mock_by_gate_set: dict[str, MagicMock] = {}
    for sensor in machine.sensor_dots.values():
        gate_set_id = sensor.voltage_sequence.gate_set.name
        if gate_set_id in mock_by_gate_set:
            continue
        mock_dc = MagicMock()
        mock_dc.get_voltage.return_value = 0.0
        mock_dc.set_voltages = MagicMock()
        mock_by_gate_set[gate_set_id] = mock_dc
    machine.virtual_dc_sets = mock_by_gate_set

    ensure_quam_config_stub(machine)
    from quam_config import Quam

    with (
        patch.object(Quam, "load", return_value=machine),
        patch_action_manager_register_only(),
    ):
        node = reimport_node_to_register_actions(NODE_NAME, CALIBRATION_LIBRARY_ROOT)
        if node is None:
            node = load_library_node(NODE_NAME, CALIBRATION_LIBRARY_ROOT)

    node.machine = machine
    apply_param_overrides(
        node,
        {
            "simulate": False,
            "use_simulated_data": False,
            "num_shots": 10,
            "offset_min": -0.2,
            "offset_max": 0.2,
            "offset_step": 0.01,
            "peak_fit_side": "left",
            **param_overrides,
        },
    )

    from calibration_utils.common_utils.experiment import get_sensors

    node.namespace["sensors"] = get_sensors(node)
    node.results["ds_raw"] = generate_simulated_dataset(node)
    for sensor in node.namespace["sensors"]:
        node.namespace[f"{sensor.name}_dac_offset"] = DAC_OFFSET_V

    call_node_action(node, "analyse_data")
    call_node_action(node, "plot_data")
    if "fit_results" in node.results:
        call_node_action(node, "update_state")

    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save = make_save_analysis_plot()
    for name, figure in (node.results.get("figures") or {}).items():
        if figure is not None:
            save(figure, artifacts_dir, f"{name}.png")

    return node


@pytest.mark.analysis
def test_03b_sensor_gate_sweep_dac_analysis_and_plot(minimal_quam_factory):
    """Fit all sensors and apply the DAC state update via mocked VirtualDCSet."""
    machine = minimal_quam_factory()
    gate_set_id = machine.sensor_dots[SENSOR_NAME].voltage_sequence.gate_set.name

    node = _run_03b_analysis(machine=machine, param_overrides={})

    sensor_names = [s.name for s in node.namespace["sensors"]]
    for sensor_name in sensor_names:
        fit = node.results["fit_results"][sensor_name]
        assert fit["success"], f"Expected successful fit for {sensor_name}, got: {fit}"
        assert np.isfinite(fit["optimal_bias"])
        assert np.isfinite(fit["lorentzian_gamma"]) and fit["lorentzian_gamma"] > 0

    figures = node.results.get("figures")
    assert isinstance(figures, dict)
    assert {"phase", "amplitude_gradient"}.issubset(figures.keys())
    assert isinstance(figures["phase"], Figure)
    assert isinstance(figures["amplitude_gradient"], Figure)

    mock_dc = machine.virtual_dc_sets[gate_set_id]
    mock_dc.set_voltages.assert_called()
    last_call = mock_dc.set_voltages.call_args[0][0]
    assert SENSOR_NAME in last_call
    expected = node.results["fit_results"][SENSOR_NAME]["optimal_bias"] + DAC_OFFSET_V
    assert last_call[SENSOR_NAME] == pytest.approx(expected)
