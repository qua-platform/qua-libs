"""Fixtures and helpers for gate virtualization analysis tests using qarray.

These tests generate synthetic 2D scan datasets via the qarray
``ChargeSensedDotArray`` model and then run the node's ``analyse_data``,
``plot_data``, and ``update_state`` actions against that data — without
requiring a real QOP or QDAC connection.
"""

from __future__ import annotations

import os
import sys
import warnings
from functools import wraps
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

CURRENT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = CURRENT_DIR.parents[3]  # tests/analysis/

_cache_base = ANALYSIS_ROOT / ".pytest_cache"
_cache_base.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_base / "matplotlib"))
os.environ.setdefault("QUALIBRATE_LOG_DIR", str(_cache_base / "qualibrate"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .....path_utils import find_repo_root  # noqa: E402

REPO_ROOT = find_repo_root(CURRENT_DIR)
CALIBRATION_LIBRARY_ROOT = REPO_ROOT / "qualibration_graphs" / "quantum_dots" / "calibrations" / "gate_virtualization"
ARTIFACTS_BASE = ANALYSIS_ROOT / "artifacts"

_QUANTUM_DOTS_DIR = REPO_ROOT / "qualibration_graphs" / "quantum_dots"
if str(_QUANTUM_DOTS_DIR) not in sys.path:
    sys.path.insert(0, str(_QUANTUM_DOTS_DIR))

from validation_utils.charge_stability.default import init_dot_model  # noqa: E402
from .quam_factory import create_gate_virtualization_quam  # noqa: E402


# =============================================================================
# qarray model fixtures
# =============================================================================


@pytest.fixture
def dot_model():
    """Return a fully configured qarray ChargeSensedDotArray (6 dots + 1 sensor)."""
    return init_dot_model()


# =============================================================================
# Simulation helpers
# =============================================================================

from .simulation_helpers import simulate_sensor_device_scan, sweep_voltages_mV  # noqa: E402


def simulate_plunger_plunger_scan(
    model,
    v_plunger_x: np.ndarray,
    v_plunger_y: np.ndarray,
    plunger_x_gate_idx: int = 0,
    plunger_y_gate_idx: int = 1,
    *,
    sensor_gate_idx: int = 6,
    sensor_operating_point: float = 15.0,
    base_voltages: Optional[np.ndarray] = None,
    sensor_compensation: Optional[Dict[int, float]] = None,
) -> xr.Dataset:
    """Simulate a 2D plunger-plunger charge stability scan using qarray.

    Parameters
    ----------
    model : ChargeSensedDotArray
        A configured qarray model.
    v_plunger_x : np.ndarray
        1-D voltages for plunger X (in mV, length M).
    v_plunger_y : np.ndarray
        1-D voltages for plunger Y (in mV, length N).
    plunger_x_gate_idx : int
        Gate index for plunger X (default 0).
    plunger_y_gate_idx : int
        Gate index for plunger Y (default 1).
    sensor_gate_idx : int
        Gate index for the sensor (default 6).
    sensor_operating_point : float
        Constant voltage on the sensor gate (mV).
    base_voltages : np.ndarray, optional
        Base voltage vector (length = n_gates).  Defaults to zeros.
    sensor_compensation : dict, optional
        Map of ``{plunger_gate_idx: alpha}`` coefficients.  When provided,
        the sensor voltage is adjusted as
        ``sensor_op + sum(alpha_i * (v_i - center_i))`` to keep the sensor
        on its Coulomb peak as plunger voltages change.

    Returns
    -------
    xr.Dataset
        Dataset with ``I`` and ``Q`` variables, dimensions
        ``(sensors, y_volts, x_volts)`` — matching the node's raw format.
        ``x_volts`` = plunger X, ``y_volts`` = plunger Y.
    """
    n_gates = max(plunger_x_gate_idx, plunger_y_gate_idx, sensor_gate_idx) + 1
    if base_voltages is None:
        base_voltages = np.zeros(n_gates)

    cx = (v_plunger_x[0] + v_plunger_x[-1]) / 2
    cy = (v_plunger_y[0] + v_plunger_y[-1]) / 2

    rows = []
    for vy in v_plunger_y:
        for vx in v_plunger_x:
            v = base_voltages.copy()
            v[plunger_x_gate_idx] = vx
            v[plunger_y_gate_idx] = vy
            s_v = sensor_operating_point
            if sensor_compensation:
                alpha_x = sensor_compensation.get(plunger_x_gate_idx, 0.0)
                alpha_y = sensor_compensation.get(plunger_y_gate_idx, 0.0)
                s_v += alpha_x * (vx - cx) + alpha_y * (vy - cy)
            v[sensor_gate_idx] = s_v
            rows.append(v)

    voltage_array = np.array(rows)
    z, _ = model.charge_sensor_open(-voltage_array)
    z = z.squeeze()
    signal_2d = z.reshape(len(v_plunger_y), len(v_plunger_x))

    v_x_V = v_plunger_x * 1e-3
    v_y_V = v_plunger_y * 1e-3

    I_data = signal_2d[np.newaxis, :, :]
    Q_data = np.zeros_like(I_data)

    ds = xr.Dataset(
        {
            "I": xr.DataArray(
                I_data,
                dims=["sensors", "y_volts", "x_volts"],
                coords={
                    "sensors": ["sensor_1"],
                    "x_volts": v_x_V,
                    "y_volts": v_y_V,
                },
            ),
            "Q": xr.DataArray(
                Q_data,
                dims=["sensors", "y_volts", "x_volts"],
                coords={
                    "sensors": ["sensor_1"],
                    "x_volts": v_x_V,
                    "y_volts": v_y_V,
                },
            ),
        }
    )
    return ds


def simulate_sensor_sweep(
    model,
    v_sensor_mV: np.ndarray,
    *,
    sensor_gate_idx: int = 6,
    base_voltages: Optional[np.ndarray] = None,
) -> xr.Dataset:
    """Simulate a 1D sensor gate sweep with other gates at *base_voltages*.

    Parameters
    ----------
    model : ChargeSensedDotArray
        A configured qarray model.
    v_sensor_mV : np.ndarray
        1-D voltages for the sensor gate (in mV).
    sensor_gate_idx : int
        Gate index for the sensor (default 6).
    base_voltages : np.ndarray, optional
        Base voltage vector (length = n_gates).  Non-sensor entries set the
        plunger/barrier voltages during the sweep.  Defaults to zeros.
    """
    n_gates = sensor_gate_idx + 1
    if base_voltages is None:
        base_voltages = np.zeros(n_gates)

    voltage_array = np.tile(base_voltages, (len(v_sensor_mV), 1))
    voltage_array[:, sensor_gate_idx] = v_sensor_mV

    z, _ = model.charge_sensor_open(-voltage_array)
    z = z.squeeze()

    v_V = v_sensor_mV * 1e-3
    I_data = z[np.newaxis, :]
    Q_data = np.zeros_like(I_data)

    return xr.Dataset(
        {
            "I": xr.DataArray(
                I_data,
                dims=["sensors", "x_volts"],
                coords={"sensors": ["sensor_1"], "x_volts": v_V},
            ),
            "Q": xr.DataArray(
                Q_data,
                dims=["sensors", "x_volts"],
                coords={"sensors": ["sensor_1"], "x_volts": v_V},
            ),
        }
    )


# =============================================================================
# Node loading helpers
# =============================================================================


def _reimport_node_to_register_actions(node_name: str, library_root: Path) -> Any | None:
    """Re-import the node module and return the node with registered actions.

    Library scanning uses inspection mode and stops before decorators run, so
    the scanned node has no registered actions. Re-importing with ActionManager
    patched to register-only produces a node with analyse_data, plot_data, etc.
    """
    node_file = library_root / f"{node_name}.py"
    if not node_file.exists():
        return None
    mod_name = f"_analysis_node_{node_name}"
    spec = spec_from_file_location(mod_name, node_file)
    if spec is None or spec.loader is None:
        return None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "node", None)


def _patch_action_manager_register_only():
    """Patch ActionManager.register_action to only register, not execute at import.

    The default decorator runs the action immediately when skip_if is False,
    which would execute create_qua_program etc. during module load. For
    analysis tests we only want to register actions so we can call them
    explicitly later.
    """
    from qualibrate.runnables.run_action.action import Action
    from qualibrate.runnables.run_action.action_manager import ActionManager

    def _register_only(
        self,
        node: Any,
        func: Any = None,
        *,
        skip_if: bool = False,
    ) -> Any:
        def decorator(f: Any) -> Any:
            action = Action(f, self)
            action_name = f.__name__
            self.actions[action_name] = action

            @wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.run_action(action_name, node, *args, **kwargs)

            return wrapper

        return decorator if func is None else decorator(func)

    return patch.object(ActionManager, "register_action", _register_only)


def _call_node_action(node: Any, action_name: str) -> None:
    """Call a node's registered run_action by function name."""
    action_manager = getattr(node, "_action_manager", None)
    if action_manager is not None:
        actions = getattr(action_manager, "actions", {})
        action = actions.get(action_name)
        if action is not None:
            try:
                action.execute_run_action(node)
            except Exception as exc:
                warnings.warn(f"Action '{action_name}' raised: {exc}")
            return
    pytest.fail(f"Node {getattr(node, 'name', '?')} has no registered run_action '{action_name}'.")


def _apply_param_overrides(node: Any, overrides: Optional[Dict[str, Any]]) -> None:
    if not overrides:
        return
    params = getattr(node, "parameters", None)
    if params is None:
        return
    for key, value in overrides.items():
        if hasattr(params, key):
            setattr(params, key, value)


def _get_parameters_dict(node: Any) -> Dict[str, Any]:
    params_obj = getattr(node, "parameters", None)
    params: Dict[str, Any] = {}
    if params_obj is None:
        return params
    for name in dir(params_obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(params_obj, name)
            if not callable(val):
                params[name] = val
        except Exception:  # pylint: disable=broad-except
            pass
    return params


# =============================================================================
# QuAM factory fixture
# =============================================================================


@pytest.fixture
def minimal_quam_factory():
    """Factory fixture that creates a minimal gate-virtualization QuAM."""

    def _factory():
        return create_gate_virtualization_quam()

    return _factory


# =============================================================================
# Analysis runner fixture
# =============================================================================


@pytest.fixture
def save_analysis_plot():
    """Save a matplotlib figure to the artifacts directory."""

    def _save(fig: Any, artifacts_dir: Path, filename: str = "simulation.png") -> Path:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifacts_dir / filename
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    return _save


@pytest.fixture
def analysis_runner(minimal_quam_factory, save_analysis_plot):
    """Run an analysis e2e test for a gate virtualization node.

    Loads the real node from the calibration library with all decorators
    registered (but not executed), injects simulated ``ds_raw_all``, then
    runs ``analyse_data``, ``plot_data``, ``update_state``, and
    ``save_results``.

    Parameters
    ----------
    node_name : str
        Calibration node filename (e.g. ``"01_sensor_gate_compensation"``).
    ds_raw_all : dict[str, xr.Dataset]
        Simulated raw datasets keyed by ``"<sensor>_vs_<device>"``.
    param_overrides : dict, optional
        Override node parameters before running.
    artifacts_subdir : str, optional
        Override the artifact sub-directory name.

    Returns
    -------
    node
        The node after running the analysis pipeline.
    """

    def _run(
        node_name: str,
        ds_raw_all: Dict[str, xr.Dataset],
        *,
        param_overrides: Optional[Dict[str, Any]] = None,
        artifacts_subdir: Optional[str] = None,
    ) -> Any:
        from quam_config import Quam

        machine = minimal_quam_factory()

        with (
            patch.object(Quam, "load", return_value=machine),
            _patch_action_manager_register_only(),
        ):
            node = _reimport_node_to_register_actions(node_name, CALIBRATION_LIBRARY_ROOT)
        if node is None:
            pytest.fail(f"Could not load node '{node_name}' from {CALIBRATION_LIBRARY_ROOT}")

        node.machine = machine

        overrides = dict(param_overrides) if param_overrides else {}
        overrides["simulate"] = False
        _apply_param_overrides(node, overrides)

        node.results["ds_raw_all"] = ds_raw_all

        _call_node_action(node, "analyse_data")
        _call_node_action(node, "plot_data")
        _call_node_action(node, "update_state")
        _call_node_action(node, "save_results")

        artifacts_dir = ARTIFACTS_BASE / (artifacts_subdir or node_name)
        for fig in node.results.get("figures", {}).values():
            save_analysis_plot(fig, artifacts_dir)
            break  # save first figure as representative artifact

        return node

    return _run


# =============================================================================
# Pytest Hooks
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "analysis: mark test as an analysis test using qarray simulation",
    )
